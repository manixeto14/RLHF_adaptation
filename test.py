import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO
from src.env.Reward import get_reward
from collections import Counter

# Configuration
TEST_DATA_PATH = 'data/cooked_test.csv'
# You can change this path to point to the model you want to test
MODEL_PATH = 'results/Builder_PPO/models/ppo_builder_run4.zip'

if not os.path.exists(TEST_DATA_PATH):
    print(f"Error: Dataset not found at {TEST_DATA_PATH}")
    exit(1)

df_test = pd.read_csv(TEST_DATA_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Please update the MODEL_PATH variable.")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")
try:
    # We attempt to load it as DQN first, as it was trained with train_builder_dqn.py
    model = DQN.load(MODEL_PATH)
    print("Successfully loaded DQN model.")
except Exception:
    try:
        # Fallback to PPO in case you train a PPO version later
        model = PPO.load(MODEL_PATH)
        print("Successfully loaded PPO model.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

# Metrics to track to understand model failures and successes
mix_found_anywhere = 0
mix_found_top_1 = 0
mix_found_top_4 = 0
add_correct = 0
cont_correct = 0
total = len(df_test)
total_reward = 0
illegal_actions_count = 0
duplicate_penalties = 0

# Track uniqueness
unique_canvases = set()
unique_mixtures = set()

print(f"Starting evaluation on {total} test samples...")

# Iterate through each test case
for idx, row in df_test.iterrows():
    # Initialize the builder environment state for the episode
    canvas = np.full(18, -1.0, dtype=np.float32)
    build_step = 0
    fatigue = 0.0
    
    # Extract shift (robust parsing as in builder_env)
    if 'shift_encoded' in row:
        shift_val = int(row['shift_encoded'])
    else:
        if isinstance(row['shift'], str):
            shift_val = 1 if row['shift'] in ['t2', 'out'] else 0
        else:
            shift_val = int(row['shift'])
            
    selection = [row['encoded_mixture'], row['additive'], row['encoded_container']]
    
    # 18 sequential steps per episode
    for step in range(18):
        # Create the Dictionary observation
        obs = {
            'Time': np.array([row['hora_decimal']], dtype=np.float32),
            'User': int(row['encoded_user']),
            'shift': shift_val,
            'dayofweek': int(row['initdayofweek']),
            'canvas': np.copy(canvas),
            'build_step': np.array([build_step], dtype=np.float32),
            'fatigue_level': np.array([fatigue], dtype=np.float32)
        }
        
        # Predict next action
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        # Check if the model predicted an illegal action for the current step
        is_illegal = False
        if build_step < 16 and action > 16: is_illegal = True
        elif build_step == 16 and action > 5: is_illegal = True
        elif build_step == 17 and action > 1: is_illegal = True
        
        if is_illegal:
            illegal_actions_count += 1
            
        # Apply the action
        canvas[build_step] = float(action)
        build_step += 1
        fatigue = min(1.0, fatigue + 0.05)
    # Episode finished. Evaluate the final canvas against the true user selection
    target_mix = selection[0]
    target_add = selection[1]
    target_cont = selection[2]
    
    predicted_mixes = canvas[:16]
    predicted_add = canvas[16]
    predicted_cont = canvas[17]
    
    # Evaluate Mixture placement
    if target_mix in predicted_mixes:
        mix_found_anywhere += 1
        # Find the first occurrence (highest rank)
        pos = np.where(predicted_mixes == target_mix)[0][0]
        if pos == 0:
            mix_found_top_1 += 1
        if pos < 4:
            mix_found_top_4 += 1
            
    # Evaluate Additive and Container accuracy
    if predicted_add == target_add:
        add_correct += 1
    if predicted_cont == target_cont:
        cont_correct += 1
        
    # Analyze duplicate mixture suggestions (excluding mix 16 which means 'hidden/empty')
    counts = Counter(predicted_mixes)
    for m, c in counts.items():
        if m != 16 and c > 1:
            duplicate_penalties += (c - 1)
            
    # Compute the episode reward using the real environment reward function
    reward = get_reward(selection, canvas)
    total_reward += reward

    # Track unique outputs
    unique_canvases.add(tuple(canvas))
    unique_mixtures.add(tuple(predicted_mixes))

print(canvas)
print("\n" + "="*50)
print(" "*15 + "EVALUATION RESULTS")
print("="*50)
print(f"Total Samples Tested: {total}")
print(f"Average Reward per Episode: {total_reward/total:.2f}")
print("-" * 50)
print(f"Behaviors & Errors:")
print(f"  - Total Illegal Actions Taken: {illegal_actions_count} (across {total * 18} total steps)")
print(f"  - Average Duplicated Mixtures per UI: {duplicate_penalties/total:.2f} (excluding hidden slots)")
print(f"  - Unique Full Canvases Generated: {len(unique_canvases)} / {total} ({len(unique_canvases)/total*100:.2f}%)")
print(f"  - Unique Mixture Combinations: {len(unique_mixtures)} / {total} ({len(unique_mixtures)/total*100:.2f}%)")
print("-" * 50)
print(f"Mixture Performance (Did the UI show the user's preferred mix?):")
print(f"  - In Top 1 slot (Primary):    {mix_found_top_1/total*100:.2f}%")
print(f"  - In Top 4 slots (Prominent): {mix_found_top_4/total*100:.2f}%")
print(f"  - Anywhere in the 16 slots:   {mix_found_anywhere/total*100:.2f}%")
print("-" * 50)
print(f"Additive Accuracy (Exact Match):  {add_correct/total*100:.2f}%")
print(f"Container Accuracy (Exact Match): {cont_correct/total*100:.2f}%")
print("="*50)