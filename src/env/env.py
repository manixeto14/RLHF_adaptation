import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box, Discrete, Dict
import numpy as np
from .Reward import get_reward


class InterfaceEnv(gym.Env):
    
    def __init__(self, df):
        super(InterfaceEnv, self).__init__()
        self.num_mix_slots = 16 
        
        self.action_space = MultiDiscrete([self.num_mix_slots + 1] * self.num_mix_slots + [6, 2])        

        obs_dict = {f'Time': Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32),
            'User': Discrete(100),
            'shift': Discrete(2),
            'dayofweek': Discrete(7)
        }
        self.observation_space = Dict(obs_dict)
        self.state = None
        self.max_steps = 1
        self.df = df
######################################################################
    def _state_to_obs(self, state):
        obs = {
            'Time': np.array([state[-4]], dtype=np.float32),
            'User': int(state[-3]),
            'shift': int(state[-2][1])-1,
            'dayofweek': int(state[-1])

        }
        return obs
    
    def get_initial_state(self):
        sample = self.df[['encoded_user', 'hora_decimal', 'shift', 'initdayofweek']].sample(1)
        initepoch = sample.iloc[0, 1]
        user = sample.iloc[0, 0]
        shift = sample.iloc[0, 2]
        dayofweek = sample.iloc[0, 3]
        initial_mixes = list(range(self.num_mix_slots))
        if shift == 'out':
            shift = 't2'
        state = np.array(initial_mixes + [3, 0, initepoch, user, shift, dayofweek])
        return state
    
    def get_final_selection(self):
        user = int(self.state[-3])
        hora_decimal = float(self.state[-4])
        selection = list(self.df[(self.df['encoded_user'] == user) & (self.df['hora_decimal'] == hora_decimal)][['encoded_mixture','additive','encoded_container']].iloc[0])
        return selection
    
#######################################################################   
    def step(self, action):
        
        selection = self.get_final_selection()

        reward = float(get_reward(selection, action))

        new_state = np.concatenate((action, self.state[-2:]))

        self.state = new_state
        done = True

        obs =  self._state_to_obs(self.state)

        return obs, reward, done, False, {}
    


    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.state = self.get_initial_state()

        obs = self._state_to_obs(self.state)
            
        return obs, {}        
    
    def render(self):
        pass

