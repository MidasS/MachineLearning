import gym
env = gym.make("FrozenLake-v0")
observation = env.reset()

for _ in range(1000):
    env.render()
