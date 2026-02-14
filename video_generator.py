#!/usr/bin/env python3
"""
Video Dataset Generator for Super Mario RL
Records gameplay with annotated actions for training data
"""

import gymnasium as gym
import numpy as np
import torch
import cv2
try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
except ImportError:
    # Fallback for gymnasium compatibility
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
from wrappers import wrap_mario
from ppo import ActorCritic
import os
from datetime import datetime

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# Action names for annotation
ACTION_NAMES = COMPLEX_MOVEMENT

class VideoRecorder:
    def __init__(self, output_dir="videos", fps=30):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)
        
    def create_video_writer(self, filename, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filepath = os.path.join(self.output_dir, filename)
        return cv2.VideoWriter(filepath, fourcc, self.fps, frame_size)
    
    def annotate_frame(self, frame, action, reward, stage, info, episode_time):
        """Add annotations to the frame showing game state and actions"""
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame = frame
            
        # Resize for better visibility
        display_frame = cv2.resize(display_frame, (512, 480))
        
        # Add black sidebar for annotations
        sidebar = np.zeros((480, 200, 3), dtype=np.uint8)
        display_frame = np.hstack([display_frame, sidebar])
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green
        thickness = 1
        
        y_offset = 30
        line_height = 25
        
        # Title
        cv2.putText(display_frame, "MARIO AI - VIDEO DATASET", (520, y_offset), 
                   font, 0.6, (0, 255, 255), 2)
        y_offset += line_height * 2
        
        # Action annotation
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"Action {action}"
        cv2.putText(display_frame, f"Action: {action_name}", (520, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += line_height
        
        # Reward
        cv2.putText(display_frame, f"Reward: {reward:.2f}", (520, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += line_height
        
        # Stage/Level
        cv2.putText(display_frame, f"Stage: {stage}", (520, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += line_height
        
        # Time
        cv2.putText(display_frame, f"Time: {episode_time:.1f}s", (520, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += line_height
        
        # Score
        if 'score' in info:
            cv2.putText(display_frame, f"Score: {info['score']}", (520, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += line_height
        
        # Coins
        if 'coins' in info:
            cv2.putText(display_frame, f"Coins: {info['coins']}", (520, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += line_height
        
        # Lives
        if 'life' in info:
            cv2.putText(display_frame, f"Lives: {info['life']}", (520, y_offset), 
                       font, font_scale, color, thickness)
            y_offset += line_height
        
        # Progress bar for x-position
        y_offset += line_height
        if 'x_pos' in info:
            x_pos = info['x_pos']
            cv2.putText(display_frame, f"Position: {x_pos}", (520, y_offset), 
                       font, font_scale, color, thickness)
        
        # Visual indicator of action
        y_offset += line_height * 2
        cv2.putText(display_frame, "Action Visual:", (520, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        # Draw action buttons
        buttons = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B']
        button_states = self._get_button_states(action_name)
        
        for i, (btn, active) in enumerate(zip(buttons, button_states)):
            btn_color = (0, 255, 0) if active else (100, 100, 100)
            cv2.rectangle(display_frame, (520 + (i % 3) * 60, y_offset + (i // 3) * 30),
                         (570 + (i % 3) * 60, y_offset + 20 + (i // 3) * 30), btn_color, -1)
            cv2.putText(display_frame, btn, (525 + (i % 3) * 60, y_offset + 15 + (i // 3) * 30),
                       font, 0.4, (0, 0, 0), 1)
        
        return display_frame
    
    def _get_button_states(self, action_name):
        """Parse action name to determine which buttons are pressed"""
        action_str = str(action_name).upper()
        return [
            'UP' in action_str,
            'DOWN' in action_str,
            'LEFT' in action_str,
            'RIGHT' in action_str,
            'A' in action_str or 'JUMP' in action_str,
            'B' in action_str or 'RUN' in action_str or 'FIRE' in action_str
        ]


def load_model(model_path="mario_1_1_ppo.pt"):
    """Load the trained PPO model"""
    model = ActorCritic(n_frame=4, act_dim=12).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file {model_path} not found. Using random initialization.")
    
    model.eval()
    return model


def make_env():
    """Create the Mario environment"""
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    return env


def generate_video(model, duration_minutes=2, output_filename=None):
    """
    Generate gameplay video with annotations
    
    Args:
        model: Trained ActorCritic model
        duration_minutes: How long to record (in minutes)
        output_filename: Name for output video file
    """
    env = make_env()
    recorder = VideoRecorder(output_dir="videos")
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"mario_gameplay_{timestamp}.mp4"
    
    obs = env.reset()
    done = False
    
    # Initialize video writer (frame size after annotation)
    frame_size = (712, 480)  # 512 game + 200 sidebar
    video_writer = recorder.create_video_writer(output_filename, frame_size)
    
    total_reward = 0
    episode_start_time = datetime.now()
    frames_captured = 0
    max_frames = int(duration_minutes * 60 * 30)  # 30 fps
    
    print(f"Recording {duration_minutes} minutes of gameplay...")
    print(f"Output: videos/{output_filename}")
    
    while not done and frames_captured < max_frames:
        # Convert observation to tensor
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get action from model
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1).item()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Get raw frame for video (render)
        frame = env.render(mode='rgb_array')
        
        # Calculate episode time
        episode_time = (datetime.now() - episode_start_time).total_seconds()
        
        # Annotate frame
        annotated_frame = recorder.annotate_frame(
            frame, action, reward, info.get('stage', 1), info, episode_time
        )
        
        # Write to video
        video_writer.write(annotated_frame)
        frames_captured += 1
        
        # Progress update every 30 seconds of video
        if frames_captured % 900 == 0:
            progress = (frames_captured / max_frames) * 100
            print(f"Progress: {progress:.1f}% - Reward: {total_reward:.2f} - Stage: {info.get('stage', 1)}")
    
    # Release video writer
    video_writer.release()
    env.close()
    
    print(f"\nVideo saved to: videos/{output_filename}")
    print(f"Total frames: {frames_captured}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Duration: {episode_time:.1f} seconds")
    
    return output_filename


def generate_dataset(num_videos=5, duration_minutes=2):
    """
    Generate multiple gameplay videos for dataset creation
    
    Args:
        num_videos: Number of videos to generate
        duration_minutes: Duration of each video
    """
    print("=" * 60)
    print("SUPER MARIO VIDEO DATASET GENERATOR")
    print("=" * 60)
    
    # Load model
    model = load_model("mario_1_1_ppo.pt")
    print(f"Using device: {device}")
    print(f"Generating {num_videos} videos, {duration_minutes} minutes each")
    print()
    
    generated_videos = []
    
    for i in range(num_videos):
        print(f"\n--- Video {i+1}/{num_videos} ---")
        try:
            filename = generate_video(model, duration_minutes)
            generated_videos.append(filename)
        except Exception as e:
            print(f"Error generating video {i+1}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated {len(generated_videos)} videos:")
    for v in generated_videos:
        print(f"  - videos/{v}")
    
    return generated_videos


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Mario gameplay videos with annotations")
    parser.add_argument("--videos", type=int, default=1, help="Number of videos to generate")
    parser.add_argument("--duration", type=int, default=2, help="Duration per video in minutes")
    parser.add_argument("--model", type=str, default="mario_1_1_ppo.pt", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if args.videos == 1:
        # Generate single video
        generate_video(model, args.duration)
    else:
        # Generate dataset
        generate_dataset(args.videos, args.duration)
