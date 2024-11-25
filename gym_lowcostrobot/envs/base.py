import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium_robotics.utils import mujoco_utils
from gym_lowcostrobot import ASSETS_PATH, BASE_LINK_NAME

class BaseEnv(gym.Env):
    """
    ## Description

    Base env with a cube
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
        self.task = task
        self.obs_type = obs_type
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.action_mode = action_mode

        # Load the Mujoco model and data
        self.load_model_data(self.task)

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4}[action_mode]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        self.nb_dof = 6

        # Set the observations space
        observation_subspaces = {
            "agent_pos": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "agent_vel": gym.spaces.Box(low=-10.0, high=10.0, shape=(6,)),
        }
        if self.obs_type == "pixels":
            observation_subspaces["pixels"] = gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model)
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        # Set the render utilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = -75
            self.viewer.cam.distance = 1
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)

        # Set additional utils
        self.threshold_height = 0.5
        self.cube_low = np.array([-0.15, 0.10, 0.015])
        self.cube_high = np.array([0.15, 0.25, 0.015])
        self.target_low = np.array([-0.15, 0.10, 0.005])
        self.target_high = np.array([0.15, 0.25, 0.005])

        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        # if the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1

        self.control_decimation = 4 # number of simulation steps per control step

    def load_model_data(self, task):
        xml_path = os.path.join(ASSETS_PATH, f"{task}.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path, {})
        self.data = mujoco.MjData(self.model)

    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="link_6", nb_dof=6, regularization=1e-6):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :return: numpy array of target joint positions
        """
        try:
            # Get the joint ID from the name
            joint_id = self.model.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")

        # Get the current end effector position
        ee_id = self.model.body(joint_name).id
        ee_pos = self.data.geom_xpos[ee_id]

        # Compute the Jacobian
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jac, None, joint_id)

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos

        # Compute the pseudoinverse of the Jacobian with regularization
        jac_reg = jac[:, :nb_dof].T @ jac[:, :nb_dof] + regularization * np.eye(nb_dof)
        jac_pinv = np.linalg.inv(jac_reg) @ jac[:, :nb_dof].T

        # Compute target joint velocities
        qdot = jac_pinv @ delta_pos

        # Normalize joint velocities to avoid excessive movements
        qdot_norm = np.linalg.norm(qdot)
        if qdot_norm > 1.0:
            qdot /= qdot_norm

        # Read the current joint positions
        qpos = self.data.qpos[self.arm_dof_id:self.arm_dof_id+nb_dof]

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        if self.action_mode == "ee":
            # raise NotImplementedError("EE mode not implemented yet")
            ee_action, gripper_action = action[:3], action[-1]

            # Update the robot position based on the action
            ee_id = self.model.body("link_6").id
            ee_target_pos = self.data.xpos[ee_id] + ee_action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
            target_qpos[-1:] = gripper_action
        elif self.action_mode == "joint":
            target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            target_qpos = np.array(action).clip(target_low, target_high)
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()

    def get_observation(self):
        # qpos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, q6, gripper]
        # qvel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dq6, dgripper]
        observation = {
            "agent_pos": self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof].astype(np.float32),
            "agent_vel": self.data.qvel[self.arm_dof_vel_id:self.arm_dof_vel_id+self.nb_dof].astype(np.float32),
        }
        if self.obs_type == "pixels":
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["pixels"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = robot_qpos

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the cube and the distance between the end effector and the cube
        reward = self.compute_reward()
        return observation, reward, self.is_done(), False, {}
        # Obs, reward, done, trucated, info

    def compute_reward(self):
        raise NotImplementedError() 
    
    def is_done(self):
        raise NotImplementedError()

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.obs_type == "pixels":
            self.renderer._gl_context.__del__()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer._gl_context.__del__()
