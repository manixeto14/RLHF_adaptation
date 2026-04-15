import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
from src.env.Reward import get_reward

class BuilderInterfaceEnv(gym.Env):
    """
    Sequential (Builder) environment for interface adaptation.
    
    This environment frames the UI design as a constructive process.
    Instead of outputting the entire UI in a single step (Contextual Bandit approach),
    the agent must build it over 18 sequential steps per episode.
    
    Steps:
    - Steps 0-15: Choose the mixture component for slots 0 to 15. Valid actions: 0-16.
    - Step 16: Choose the additive level. Valid actions: 0-5.
    - Step 17: Choose the container status. Valid actions: 0-1.
    """
    
    def __init__(self, df):
        """
        Initialize the environment.
        Args:
            df (pd.DataFrame): Training data containing user interactions.
        """
        super(BuilderInterfaceEnv, self).__init__()
        self.num_mix_slots = 16 
        
        # Max options in any given step is 17 (for mixtures: indices 0-16).
        self.action_space = Discrete(17)

        # Observation space extends user context with the UI canvas and the build step.
        obs_dict = {
            'Time': Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32),
            'User': Discrete(100),
            'shift': Discrete(2),
            'dayofweek': Discrete(7),
            'canvas': Box(low=-1.0, high=17.0, shape=(18,), dtype=np.float32),
            'build_step': Box(low=0.0, high=18.0, shape=(1,), dtype=np.float32),
            'fatigue_level': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        }
        self.observation_space = Dict(obs_dict)
        
        self.max_steps_per_episode = 18
        self.df = df
        
        self.canvas = None
        self.build_step = 0
        self.user_data = None
        self.fatigue = 0.0

    def _get_obs(self):
        """Return the current dictionary observation."""
        obs = {
            'Time': np.array([self.user_data['hora_decimal']], dtype=np.float32),
            'User': int(self.user_data['encoded_user']),
            'shift': int(self.user_data['shift_encoded']) if 'shift_encoded' in self.user_data else int(self.user_data['shift'] == 't2' or self.user_data['shift'] == 'out' or self.user_data['shift'] == 1), 
            # Note: shift parsing depends on dataset exact values. Fallback generic parsing.
            'dayofweek': int(self.user_data['initdayofweek']),
            'canvas': np.copy(self.canvas),
            'build_step': np.array([self.build_step], dtype=np.float32),
            'fatigue_level': np.array([self.fatigue], dtype=np.float32)
        }
        # Refine shift if it's string. The previous implementation mapped shift -> int
        if isinstance(self.user_data['shift'], str):
            shift_val = 1 if self.user_data['shift'] in ['t2', 'out'] else 0
        else:
            shift_val = int(self.user_data['shift'])
        # Handle string dayofweek just in case, logic from initial env.py relies on integer.
        obs['shift'] = shift_val

        return obs

    def get_initial_state(self):
        """Samples a random user interaction from the dataset."""
        sample = self.df[['encoded_user', 'hora_decimal', 'shift', 'initdayofweek']].sample(1)
        return sample.iloc[0]

    def get_final_selection(self):
        """Returns the real preferred interface for the current user and time."""
        user = int(self.user_data['encoded_user'])
        hora_decimal = float(self.user_data['hora_decimal'])
        selection = list(self.df[(self.df['encoded_user'] == user) & (self.df['hora_decimal'] == hora_decimal)][['encoded_mixture','additive','encoded_container']].iloc[0])
        return selection

    def reset(self, seed=None):
        """
        Resets the environment for a new episode.
        """
        super().reset(seed=seed)
        
        self.user_data = self.get_initial_state()
        
        # Start with a canvas filled with -1 (empty)
        self.canvas = np.full(18, -1.0, dtype=np.float32)
        self.build_step = 0
        self.fatigue = 0.0 # Starts without fatigue
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Process the action for the current step of building the UI.
        """
        action = int(action)
        reward = 0.0
        terminated = False
        
        # Check for illegal actions and penalize harshly
        is_illegal = False
        if self.build_step < 16:  # Mixture slots (0-16 valid)
            if action > 16: is_illegal = True
        elif self.build_step == 16: # Additive (0-5 valid)
            if action > 5: is_illegal = True
        elif self.build_step == 17: # Container (0-1 valid)
            if action > 1: is_illegal = True

        if is_illegal:
            # Per user request: DO NOT clip, just give a very bad reward.
            # We record the bad action in the canvas, and issue a harsh immediate penalty.
            reward -= 150.0 
            
        # Apply action to canvas
        self.canvas[self.build_step] = float(action)
        self.build_step += 1
        
        # Increase user fatigue slightly each step
        self.fatigue = min(1.0, self.fatigue + 0.05)

        # Reached the end of the building process
        if self.build_step == 18:
            terminated = True
            
            # Use original reward logic on the constructed UI
            selection = self.get_final_selection()
            # The get_reward function expects state = [mix1... mix16, additive, container]
            final_reward = float(get_reward(selection, self.canvas))
            
            # Deliver the final composition reward (Sparse Reward)
            reward += final_reward
            
        obs = self._get_obs()

        return obs, reward, terminated, False, {}

    def render(self):
        pass
