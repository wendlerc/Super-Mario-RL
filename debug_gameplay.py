#!/usr/bin/env python3
"""
Debug script to understand why Mario stops at 4s
"""

import torch
import numpy as np
from ppo import ActorCritic, make_env

device = "cpu"

def debug_gameplay():
    """Debug a single episode to see when/why Mario stops"""
    
    # Load model
    model = ActorCritic(n_frame=4, act_dim=12).to(device)
    model.load_state_dict(torch.load("mario_1_1_ppo.pt", map_location=device))
    model.eval()
    
    # Create environment
    env = make_env()
    obs = env.reset()
    
    done = False
    steps = 0
    total_reward = 0
    
    print("="*60)
    print("DEBUG: Mario Gameplay")
    print("="*60)
    
    while not done and steps < 500:  # Max 500 steps for debug
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1).item()
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print debug info every 10 steps
        if steps % 10 == 0 or done:
            print(f"Step {steps:3d}: Reward={reward:6.2f}, Total={total_reward:8.2f}, "
                  f"Done={done}, Stage={info.get('stage', 1)}, "
                  f"X={info.get('x_pos', 0)}")
        
        if done:
            print(f"\n>>> MARIO DIED at step {steps}")
            print(f">>> Reason: done=True triggered")
            print(f">>> Info: {info}")
            break
    
    env.close()
    
    print("\n" + "="*60)
    print(f"Summary: {steps} steps, Total reward: {total_reward:.2f}")
    print("="*60)

if __name__ == "__main__":
    debug_gameplay()
