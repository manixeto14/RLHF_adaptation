import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.env import InterfaceEnv
import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


# Callback to save the model every N episodes
# class EpisodeCheckpointCallback(BaseCallback):
#     def _init_(self, save_every_episodes, rollout_steps, save_path, name_prefix='ppo_episode_prueba', verbose=0):
#         super()._init_(verbose)
#         self.save_every_episodes = save_every_episodes
#         self.rollout_steps = rollout_steps
#         self.save_path = save_path
#         self.name_prefix = name_prefix
#         self.episode_counter = 0

#     def _on_step(self) -> bool:
#         if self.n_calls % self.rollout_steps == 0:
#             self.episode_counter += 1
#             if self.episode_counter % self.save_every_episodes == 0:
#                 save_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.episode_counter}_episodes")
#                 self.model.save(save_file)
#                 if self.verbose > 0:
#                     print(f"✅ Modelo guardado en: {save_file}")
#         return True

# # Callback to log rewards into Tensorboard
# class TensorboardRewardCallback(BaseCallback):
#     def __init__(self, verbose=1,  save_every_episodes, rollout_steps, save_path, name_prefix='PPO_save'):
#         super(TensorboardRewardCallback, self).__init__(verbose)
#         self.episode_rewards = []
#         self.current_rewards = 0

#         self.save_every_episodes = save_every_episodes
#         self.rollout_steps = rollout_steps
#         self.save_path = save_path
#         self.name_prefix = name_prefix
#         self.episode_counter = 0

#     def _on_step(self) -> bool:
#         # Actual reward
#         self.current_rewards += self.locals["rewards"][0]

#         # Save reward if episode finished
#         if self.locals["dones"][0]:
#             self.episode_rewards.append(self.current_rewards)
            
#             # Log into Tensorboard
#             mean_reward = np.mean(self.episode_rewards[-10:])  # 10 episode smoothing
            
#             self.logger.record("rollout/average_reward_per_episode", mean_reward)

#             # Reset for next episode
#             self.current_rewards = 0
#             self.current_length = 0


#             if self.n_calls % self.rollout_steps == 0:
#                 self.episode_counter += 1
#                 if self.episode_counter % self.save_every_episodes == 0:
#                     save_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.episode_counter}_episodes")
#                     self.model.save(save_file)
#                     if self.verbose > 0:
#                         print(f"✅ Modelo guardado en: {save_file}")
#         return True
    

class RewardCheckpointCallback(BaseCallback):
    def __init__(self, save_every_episodes, rollout_steps, save_path, name_prefix='PPO_save', verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

        self.save_every_episodes = save_every_episodes
        self.rollout_steps = rollout_steps
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.checkpoint_internal_counter = 0

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            
            if len(self.episode_rewards) > 0:
                smoothing_window = min(10, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-smoothing_window:])
                self.logger.record("rollout/average_reward_per_episode", mean_reward)

            self.current_rewards = 0
        
        if self.rollout_steps > 0 and self.n_calls > 0 and self.n_calls % self.rollout_steps == 0:
            self.checkpoint_internal_counter += 1
            if self.save_every_episodes > 0 and self.checkpoint_internal_counter % self.save_every_episodes == 0:
                if self.save_path:
                    save_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.checkpoint_internal_counter}_episodes")
                    self.model.save(save_file)
                    if self.verbose > 0:
                        print(f"✅ Model saved in: {save_file}")
        return True



# Algoritmo en uso para organizar los resultados
alg_name = "PPO"
models_dir = f"results/{alg_name}/models/"
logs_dir = f"results/{alg_name}/logs/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

df = pd.read_csv('data/cooked_train.csv')

env = InterfaceEnv(df)

# Check environment
check_env(env, warn=True)

# Vectorize the environment
env = DummyVecEnv([lambda: env])

name = 'new_data_all1'

callback = RewardCheckpointCallback(    
    save_every_episodes=20000,
    rollout_steps=1,
    save_path=models_dir,
    name_prefix=name,
    verbose=1
)

try:
    model = PPO.load(os.path.join(models_dir, name), env=env)
    model.learn(total_timesteps=20000, callback=callback, reset_num_timesteps=False)

except:
    model = PPO("MultiInputPolicy", env, n_steps=1024, verbose=1, tensorboard_log=logs_dir)
    model.learn(total_timesteps=150000, callback=callback, reset_num_timesteps=True)

model.save(os.path.join(models_dir, name))
