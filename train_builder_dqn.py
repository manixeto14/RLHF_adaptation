import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.env.builder_env import BuilderInterfaceEnv

# Callback to log rewards and save checkpoints
class RewardCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every `save_every_episodes` episodes
    and logging smoothed average reward to tensorboard.
    """
    def __init__(self, save_every_episodes, rollout_steps, save_path, name_prefix='DQN_save', verbose=1):
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
            # Increment episode counter internally for DQN (since rollout_steps behaves differently than PPO)
            self.checkpoint_internal_counter += 1
            
            if self.save_every_episodes > 0 and self.checkpoint_internal_counter % self.save_every_episodes == 0:
                if self.save_path:
                    save_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.checkpoint_internal_counter}_episodes")
                    self.model.save(save_file)
                    if self.verbose > 0:
                        print(f"✅ Model saved in: {save_file}")
                        
        return True


def main():
    alg_name = "DQN"
    models_dir = f"results/Builder_{alg_name}/models/"
    logs_dir = f"results/Builder_{alg_name}/logs/"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv('data/cooked_train.csv')

    # Initialize environment
    env = BuilderInterfaceEnv(df)

    # Validation
    print("Checking environment API compliance...")
    check_env(env, warn=True)

    # Vectorize
    env = DummyVecEnv([lambda: env])

    name = 'dqn_builder_run1'

    callback = RewardCheckpointCallback(    
        save_every_episodes=5000,
        rollout_steps=1,
        save_path=models_dir,
        name_prefix=name,
        verbose=1
    )

    # DQN requires MultiInputPolicy for Dict spaces
    try:
        model = DQN.load(os.path.join(models_dir, name), env=env)
        print("Loaded existing DQN model.")
        model.learn(total_timesteps=300_000, callback=callback, reset_num_timesteps=False)
    except Exception as e:
        print(f"Starting new DQN model. ({e})")
        # DQN hyperparams can be tweaked. Using standard ones for now.
        model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=logs_dir,
                    exploration_fraction=0.2, exploration_final_eps=0.05)
        model.learn(total_timesteps=300_000, callback=callback, reset_num_timesteps=True)

    model.save(os.path.join(models_dir, name))
    print("Training finished.")

if __name__ == '__main__':
    main()
