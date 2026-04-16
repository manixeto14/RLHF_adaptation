import pandas as pd
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box
import numpy as np
from all_obs_env import  InterfaceEnv


# df = pd.read_csv(r'C:\Users\manex\OneDrive - Mondragon Unibertsitatea\Unibertsitatea\3.kurtsoa\Lana\Mixing-machine\Python\Version_9_txukuna\RLHF_adaptation\src\Synthetic_data\cooked_train.csv')
df = pd.read_csv(r'data/df_train.csv')
env = InterfaceEnv(df)


obs, info = env.reset()
done = False
step_count = 0

while not done:
    print(f"Step: {step_count}, Obs: {obs}")

    action = env.action_space.sample()
    print(f"Selected action: {action}")

    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}\nFinal obs: {obs}")

    step_count += 1
    if done or truncated:

        print("Episode finished!")
        break
