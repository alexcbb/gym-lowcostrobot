import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments
import matplotlib.pyplot as plt


# Create the environment
env = gym.make("LiftCube-v0", render_mode="rgb_array")

# Reset the environment
observation, info = env.reset()

for _ in range(3):
    # Sample random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminted, truncated, info = env.step(action)
    plt.imshow(observation["pixels"])
    plt.show()
    # Reset the environment if it's done
    # if terminted or truncated:
    #     observation, info = env.reset()
env.close()