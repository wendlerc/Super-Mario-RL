#!/usr/bin/env python3
"""
Debug script to analyze DQN action selection
"""

import torch
import torch.nn as nn
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import wrap_mario

device = "cpu"

# Action names
ACTION_NAMES = COMPLEX_MOVEMENT

# Import DQN model and arange
def arange(s):
    """Convert observation to proper format for DQN"""
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)

class DQNModel(nn.Module):
    """DQN Model - matches duel_dqn.py exactly"""
    def __init__(self, n_frame, n_action, device):
        super(DQNModel, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)
        self.device = device
        # This seq is used in the saved model
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))
        return q

def debug_dqn():
    """Debug DQN action selection"""
    print("="*60)
    print("DQN Action Selection Debug")
    print("="*60)
    
    # Load model
    print("\nLoading DQN model...")
    q = DQNModel(n_frame=4, n_action=12, device=device).to(device)
    q.load_state_dict(torch.load("mario_q_target.pth", map_location=device))
    q.eval()
    print("✓ Model loaded")
    
    # Create environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    
    obs = env.reset()
    done = False
    
    print("\nRunning 50 steps with action logging...")
    print("-"*60)
    
    actions_selected = []
    for step in range(50):
        # Prepare input
        s = arange(obs)
        
        # Get Q-values
        with torch.no_grad():
            q_values = q(s)
        
        # Select action (argmax)
        action = q_values.argmax(dim=-1).item()
        actions_selected.append(action)
        action_name = ACTION_NAMES[action]
        
        # Show Q-values for first few steps
        if step < 5:
            print(f"\nStep {step}:")
            print(f"  Q-values: {q_values.detach().cpu().numpy().flatten()[:6]}")  # First 6 actions
            print(f"  Selected: {action} ({action_name})")
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\n>>> Died at step {step}")
            obs = env.reset()
            done = False
    
    env.close()
    
    # Analyze action distribution
    print("\n" + "="*60)
    print("Action Distribution:")
    print("-"*60)
    from collections import Counter
    action_counts = Counter(actions_selected)
    for action, count in sorted(action_counts.items()):
        action_name_str = str(ACTION_NAMES[action])
        print(f"  {action:2d} ({action_name_str:15s}): {count:3d} times ({count/len(actions_selected)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Summary:")
    if action_counts[6] > 20:  # LEFT is action 6
        print("⚠️  Model is stuck going LEFT frequently!")
    elif action_counts[0] > 20:  # NOOP
        print("⚠️  Model is mostly doing NOOP (no action)")
    else:
        print("✓ Action distribution looks varied")
    print("="*60)

if __name__ == "__main__":
    debug_dqn()
