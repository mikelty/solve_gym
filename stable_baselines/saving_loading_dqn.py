# -*- coding: utf-8 -*-
import gym
import numpy as np

from stable_baselines import DQN

"""## Create the Gym env and instantiate the agent

For this example, we will use Lunar Lander environment.

"Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine. "

Lunar Lander environment: [https://gym.openai.com/envs/LunarLander-v2/](https://gym.openai.com/envs/LunarLander-v2/)

![Lunar Lander](https://cdn-images-1.medium.com/max/960/1*f4VZPKOI0PYNWiwt0la0Rg.gif)

Note: vectorized environments allow to easily multiprocess training. In this example, we are using only one process, hence the DummyVecEnv.

We chose the MlpPolicy because input of CartPole is a feature vector, not images.

The type of action to use (discrete/continuous) will be automatically deduced from the environment action space
"""

env = gym.make('LunarLander-v2')


model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

"""We create a helper function to evaluate the agent:"""

def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)

      # Stats
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

  return mean_100ep_reward

"""Let's evaluate the un-trained agent, this should be a random agent."""

# Random Agent, before training
mean_reward_before_train = evaluate(model, num_steps=10000)

"""## Train the agent and save it

Warning: this may take a while
"""

# Train the agent
model.learn(total_timesteps=int(2e4), log_interval=10)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

"""## Load the trained agent"""

model = DQN.load("dqn_lunar")

# Evaluate the trained agent
mean_reward = evaluate(model, num_steps=10000)

