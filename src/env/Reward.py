from collections import Counter

def get_reward(selection,state):

    reward = 0
    num_mix_slots = len(state)-2
    mixes = state[:num_mix_slots]
    Mix_count = Counter(mixes)
    Hidden_mix = 16
    

#######################################################
    #Penalizations

    for mix in Mix_count:
        if Mix_count[mix] > 1:
            if mix != Hidden_mix:
                reward -= 30 #Penalize for showing the same mix more than once

    reward += Mix_count[Hidden_mix]*1 # Reward for hiding mix

    if Mix_count[Hidden_mix] == num_mix_slots:
        return -60 #Penalize for hiding all mixes

######################################################

    #Mixture
    if selection[0] not in state[0:num_mix_slots]:
        reward -= 60 #Penalize for not showing the desired mix
    else:
        for i, mix in enumerate(state[0:num_mix_slots]):
            if mix == selection[0]:
                reward += [20, 10, 5, 2][i] if i < 4 else 1 #Reward for showing desired mix
                break

    if selection[0] not in (10, 12) and selection[0] in mixes:
        reward += 20  # Extra reward for guessing less common mixes

#######################################################

    #Additive
    if state[-2] == selection[1]:    
        reward +=10 #Reward for guessing the right additive
    else:
        reward += - abs(state[-2]-selection[1]) * 2 #Penalize for guessing wrong additive

#######################################################

    #Container
    if state[-1] == selection[2]: 
        reward += 10    #Reward for guessing the right container
    else: reward += -10 #Penalize for guessing wrong container

#######################################################
    
    
    return reward 


if __name__ == "__main__":
    # Example usage
    selection = [1, 3, 0]  # Example selection
    state = [1,3, 4, 3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 3, 0]  # Example state
    reward = get_reward(selection, state)
    print(f"Reward: {reward}")