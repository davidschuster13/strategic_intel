from stable_baselines3 import PPO
import os
import numpy as np
from pathlib import Path
from strategic_intel_env_v7 import StrategicWargameEnv
from self_play_env import SelfPlayEnv

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train(iterations=12, steps_per_round=40_000):
    """Self-play training with policy pool to avoid non-stationary learning.
    
    Instead of alternating Blue/Red training (which is non-stationary),
    maintain a pool of opponent policies and sample from it for robustness.
    Each iteration, train against a random previous opponent to encourage generalization.
    """
    base_env = StrategicWargameEnv()
    
    # Initialize with models trained against random opponents
    blue_models = []
    red_models = []
    blue = None
    red = None
    
    print("Starting self-play training with policy pool...")
    
    for i in range(iterations):
        print(f"\n=== Self-Play Round {i+1}/{iterations} ===")
        
        # Sample opponent from previous models if pool exists, else random
        if red_models:
            prev_red_idx = np.random.randint(0, len(red_models))
            opponent_red = red_models[prev_red_idx]
            print(f"  Blue training against Red model #{prev_red_idx} from pool")
        else:
            opponent_red = None  # Use random until pool exists
            print(f"  Blue training against random Red")
        
        if blue_models:
            prev_blue_idx = np.random.randint(0, len(blue_models))
            opponent_blue = blue_models[prev_blue_idx]
            print(f"  Red training against Blue model #{prev_blue_idx} from pool")
        else:
            opponent_blue = None  # Use random until pool exists
            print(f"  Red training against random Blue")
        
        # Train Blue against opponent Red
        blue_env = SelfPlayEnv(base_env, "Blue", opponent_red)
        if blue is None:
            blue = PPO("MlpPolicy", blue_env, verbose=0, device="cpu")
        else:
            blue.set_env(blue_env)
        blue.learn(total_timesteps=steps_per_round)
        blue.save(f"{MODEL_DIR}/blue_{i}")
        blue_models.append(blue)
        
        # Train Red against opponent Blue
        red_env = SelfPlayEnv(base_env, "Red", opponent_blue)
        if red is None:
            red = PPO("MlpPolicy", red_env, verbose=0, device="cpu")
        else:
            red.set_env(red_env)
        red.learn(total_timesteps=steps_per_round)
        red.save(f"{MODEL_DIR}/red_{i}")
        red_models.append(red)
        
        print(f"  Policy pools: Blue={len(blue_models)}, Red={len(red_models)}")
        
    print(f"\n✅ Self-play training completed!")
    print(f"Final pools: {len(blue_models)} Blue models, {len(red_models)} Red models")
    return blue, red, blue_models, red_models

if __name__ == "__main__":
    train()
