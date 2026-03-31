import pandas as pd
from stable_baselines3 import PPO
import numpy as np


def _state_to_obs(state):
    #Function to convert the state to the observation dictionary
    obs = {
        'Time': np.array([state[-2]], dtype=np.float32),
        'User': int(state[-1])
    }
    return obs


import os

df_test = pd.read_csv('data/cooked_test.csv')
# Si el modelo está en la vieja ubicación (que ahora es data/) o en results/PPO/models:
model_path = 'results/PPO/models/new_data_all1.zip'
if not os.path.exists(model_path):
    model_path = 'data/original_PPO-2.zip'

print(f"Loading model from {model_path}...")
model = PPO.load(model_path)

mix = 0
add = 0
cont = 0
total = len(df_test)
mix_history = []
add_history = []
cont_history = []

for data in df_test[['encoded_user','hora_decimal']].values:
    action = list(model.predict(_state_to_obs(data), deterministic=True))[0]
    prediction = [action[0], action[-2], action[-1]]  # [mix, additive, container]

    mix_history.append(prediction[0])
    add_history.append(prediction[1])
    cont_history.append(prediction[2])

    selection = list(df_test[(df_test['encoded_user'] == data[0]) & (df_test['hora_decimal'] == data[1])][['encoded_mixture','additive','encoded_container']].iloc[0])
    if selection[0] == prediction[0]:
        mix += 1
    if selection[1] == prediction[1]:
        add += 1
    if selection[2] == prediction[2]:
        cont += 1

dicc_history = {'Mix': mix_history, 'Additive': add_history, 'Container': cont_history}
df_history = pd.DataFrame(dicc_history)
print(df_history.describe())
print(f'Mix score:{mix/total}, Additive score: {add/total}, Container score: {cont/total}', )#/len(df_test))