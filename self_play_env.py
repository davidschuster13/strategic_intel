import gymnasium as gym
import numpy as np

class SelfPlayEnv(gym.Env):
    """Fixed wrapper for self-play training - simplified to avoid recursion"""
    def __init__(self, base_env, agent_name: str, opponent_policy=None):
        super().__init__()
        self.base_env = base_env
        self.agent_name = agent_name
        self.opponent_policy = opponent_policy
        
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space  # Box space
        self.last_obs_dict = None
    
    def reset(self, seed=None, options=None):
        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        self.last_obs_dict = obs_dict
        # Return only this agent's observation (as a numpy array, not dict)
        return obs_dict[self.agent_name], info
    
    def step(self, action):
        # Get opponent action from policy or random
        if self.opponent_policy is None:
            opp_action = self.action_space.sample()
        else:
            # Use last cached observation for opponent
            opponent_name = "Red" if self.agent_name == "Blue" else "Blue"
            if self.last_obs_dict is not None:
                obs = self.last_obs_dict[opponent_name]
                opp_action, _ = self.opponent_policy.predict(obs, deterministic=False)
                opp_action = int(opp_action)
            else:
                opp_action = self.action_space.sample()
        
        # Build full action dict
        if self.agent_name == "Blue":
            actions = {"Blue": action, "Red": opp_action}
        else:
            actions = {"Blue": opp_action, "Red": action}
        
        # Step the environment
        obs_dict, rewards, terminated, truncated, info = self.base_env.step(actions)
        
        # Cache for next step
        self.last_obs_dict = obs_dict
        
        done = terminated if isinstance(terminated, bool) else terminated[self.agent_name]
        
        return obs_dict[self.agent_name], rewards[self.agent_name], done, done, info
