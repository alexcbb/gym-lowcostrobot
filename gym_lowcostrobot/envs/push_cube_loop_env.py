import numpy as np

from gym_lowcostrobot.envs.base import BaseEnv
import mujoco
# TODO : setup the following env later
class PushCubeLoopEnv(BaseEnv):
    """
    ## Description

    The robot has to push a cube with its end-effector between two goal positions.
    Once the cube reaches the goal position, the goal region is switched to the other side.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"agent_pos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"agent_vel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"pixels"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"agent_pos"`    | ✓         | ✓         | ✓        |
    | `"agent_vel"`    | ✓         | ✓         | ✓        |
    | `"pixels"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"cube_pos"`    |           | ✓         | ✓        |

    ## Reward

    The reward is the negative distance between the cube and the target position.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(
            self,
            task,
            obs_type="pixels",
            render_mode="rgb_array",
            observation_width=640,
            observation_height=480,
            visualization_width=640,
            visualization_height=480,
            action_mode="joint"
        ):
        super().__init__(task, obs_type, render_mode, observation_width, observation_height, visualization_width, visualization_height, action_mode)    
        # Set additional utils
        self.threshold_height = 0.5

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.cube_low = np.array([-0.15, 0.10, 0.015])
        self.cube_high = np.array([0.15, 0.25, 0.015])
        
        self.cube_size = 0.015
        self.cube_position = np.array([0.0,0.0,0.0])

        goal_region_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_1")
        goal_region_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_2")

        self.goal_region_1_center = self.model.geom_pos[goal_region_1_id]
        self.goal_region_2_center = self.model.geom_pos[goal_region_2_id]

        self.goal_region_high = self.model.geom_size[goal_region_1_id] 
        self.goal_region_high[:2] -= 0.008 # offset sampling region to keep cube within
        self.goal_region_low = self.goal_region_high * np.array([-1., -1., 1.])
        self.current_goal = 0 # 0 for first goal region , and 1 for second goal region
        self.control_decimation = 4 # number of simulation steps per control step

        self._step = 0
        # indicators for the reward

    def reset(self, seed=None, options=None):
        # Reset the robot to the initial position and sample the cube position
        cube_pos = self.np_random.uniform(self.goal_region_low, self.goal_region_high) 
        cube_pos[:2] += (1 - self.current_goal) * self.goal_region_1_center[:2] \
                      + self.current_goal * self.goal_region_2_center[:2]

        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.cube_dof_id:self.cube_dof_id+7]=np.concatenate([cube_pos, cube_rot])
        obs, _ = super().reset(seed=seed, options=options)
        
        return obs, {'timestamp': 0.0}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        reward, success = self.compute_reward()
        self._step += 1
        info = {'timestamp': self.data.time, 'success': success}

        return observation, reward, False, False, info


    def compute_reward(self):
        # Get the position of the cube and the distance between the end effector and the cube
        self.cube_position = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        overlap = self.get_cube_overlap()
        # if the intersection is above 95% consider the episode a success and switch goals:
        success = 0
        if overlap > 0.95:
            success = 1
            reward = +5
            self.current_goal = 1 - self.current_goal

        elif overlap > 0.0:
            reward = overlap - 1

        elif overlap == 0.0:
            # calculate distance to edge on y axis only
            goal_region_edge = self.goal_region_low[1] \
                               + (1 - self.current_goal) * self.goal_region_1_center[1] \
                               + self.current_goal * self.goal_region_2_center[1]
            
            distance_to_edge = np.sqrt((self.cube_position[1] - goal_region_edge)**2)
            # max distance to edge within the box is 0.16
            reward = min(max((-distance_to_edge / 0.16) - 1, -2), -1)
        return reward, success


    def get_cube_overlap(self):
        # Unpack the parameters
        x_cube, y_cube = self.cube_position[:2]
        w_cube = l_cube = self.cube_size
        
        goal_center = self.goal_region_1_center if self.current_goal == 0 else self.goal_region_2_center
        x_goal, y_goal = goal_center[:2] 
        w_goal, l_goal = self.goal_region_high[:2]
        
        # Calculate the overlap along the x-axis
        x_overlap = max(0, min(x_cube + w_cube, x_goal + w_goal) - max(x_cube - w_cube, x_goal - w_goal))
    
        # Calculate the overlap along the y-axis
        y_overlap = max(0, min(y_cube + l_cube, y_goal + l_goal) - max(y_cube - l_cube, y_goal - l_goal))
    
        # Calculate the area of the overlap region
        overlap_area = x_overlap * y_overlap
    
        # Calculate the area of the cube
        cube_area = w_cube * l_cube * 4
    
        # return the percentage overlap relative to the cube area
        return overlap_area / cube_area

