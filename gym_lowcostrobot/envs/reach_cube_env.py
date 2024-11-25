import numpy as np
from gym_lowcostrobot.envs.base import BaseEnv

class ReachCubeEnv(BaseEnv):
    """
    ## Description

    The robot has to reach a cube with its end-effector.

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

    The reward is the sum of two terms: the height of the cube above the threshold and the negative distance between the
    end effector and the cube.

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
        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]  

    def reset(self, seed=None, options=None):
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.cube_dof_id:self.cube_dof_id + 7] = np.concatenate([cube_pos, cube_rot])

        return super().reset(seed, options)  

    def compute_reward(self):
        cube_pos = self.data.xpos[2]
        ee_pos = self.data.xpos[8]
        ee_to_cube = np.linalg.norm(ee_pos - cube_pos)

        # Compute the reward
        return -ee_to_cube
    
    def is_done(self):
        return self.compute_reward() > -0.02

