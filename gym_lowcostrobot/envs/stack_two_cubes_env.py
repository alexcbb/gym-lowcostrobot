import numpy as np
from gym_lowcostrobot.envs.base import BaseEnv
from gymnasium import spaces

class StackTwoCubesEnv(BaseEnv):
    """
    ## Description

    The robot has to stack the blue cube on top of the red cube.

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
    - `"cube_red_pos"`: the position of the red cube, as (x, y, z)
    - `"cube_blue_pos"`: the position of the blue cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key               | `"image"` | `"state"` | `"both"` |
    | ----------------- | --------- | --------- | -------- |
    | `"agent_pos"`     | ✓         | ✓         | ✓        |
    | `"agent_vel"`     | ✓         | ✓         | ✓        |
    | `"pixels"`        | ✓         |           | ✓        |
    | `"image_top"`     | ✓         |           | ✓        |
    | `"cube_red_pos"`  |           | ✓         | ✓        |
    | `"cube_blue_pos"` |           | ✓         | ✓        |

    ## Reward

    The reward is the opposite of the distance between the top of the red cube and the blue cube.

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
        self.observation_space["cube_red_pos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        self.observation_space["cube_blue_pos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        # get dof addresses
        self.red_cube_dof_id = self.model.body("cube_red").dofadr[0]
        self.blue_cube_dof_id = self.model.body("cube_blue").dofadr[0] + 1

    def get_observation(self):
        observation = super().get_observation()
        observation["cube_red_pos"] = self.data.qpos[self.red_cube_dof_id:self.red_cube_dof_id+3].astype(np.float32)
        observation["cube_blue_pos"] = self.data.qpos[self.blue_cube_dof_id:self.blue_cube_dof_id+3].astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        # Reset the robot to the initial position and sample the cube position
        cube_red_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_red_rot = np.array([1.0, 0.0, 0.0, 0.0])
        cube_blue_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_blue_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.red_cube_dof_id:self.red_cube_dof_id+7] = np.concatenate([cube_red_pos, cube_red_rot])
        self.data.qpos[self.blue_cube_dof_id:self.blue_cube_dof_id+7] = np.concatenate([cube_blue_pos, cube_blue_rot])

        return super().reset(seed=seed, options=options)
    
    def compute_reward(self):   
        cube_red_pos = self.data.qpos[self.red_cube_dof_id:self.red_cube_dof_id+3]
        cube_blue_pos = self.data.qpos[self.blue_cube_dof_id:self.blue_cube_dof_id+3]
        target_pos = cube_red_pos + np.array([0.0, 0.0, 0.03])
        cube_blue_to_target = np.linalg.norm(cube_blue_pos - target_pos)
        return -cube_blue_to_target

    def is_done(self):
        reward = self.compute_reward()
        return reward > -0.01
