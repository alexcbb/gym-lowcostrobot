import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments
import matplotlib.pyplot as plt


# Create the environment
env = gym.make("PushCubeLoop-v0", task="push_cube_loop", render_mode="human", observation_width=128, observation_height=128)
observation, info = env.reset()

for _ in range(1000):
    # Sample random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminted, truncated, info = env.step(action)
    if terminted or truncated:
        observation, info = env.reset()
env.close()