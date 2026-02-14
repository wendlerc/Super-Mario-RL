#!/usr/bin/env python3
"""
Test script to verify the video generation setup
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("✗ PyTorch not installed")
    
    try:
        import gym
        print(f"✓ Gym {gym.__version__}")
    except ImportError:
        missing.append("gym")
        print("✗ Gym not installed")
    
    try:
        import gym_super_mario_bros
        print("✓ Gym Super Mario Bros")
    except ImportError:
        missing.append("gym-super-mario-bros")
        print("✗ Gym Super Mario Bros not installed")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
        print("✗ OpenCV not installed")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        print("✗ NumPy not installed")
    
    try:
        import nes_py
        print("✓ NES Py")
    except ImportError:
        missing.append("nes-py")
        print("✗ NES Py not installed")
    
    if missing:
        print("\n❌ Missing dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: bash setup.sh")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_model_files():
    """Check if model files are available"""
    import os
    
    models = [
        "mario_1_1_ppo.pt",
        "mario_q_target.pth"
    ]
    
    print("\nChecking model files...")
    found = False
    for model in models:
        if os.path.exists(model):
            size_mb = os.path.getsize(model) / (1024 * 1024)
            print(f"✓ {model} ({size_mb:.1f} MB)")
            found = True
        else:
            print(f"✗ {model} not found")
    
    if not found:
        print("\n⚠️  No pre-trained models found.")
        print("You can still run the script but it will use random initialization.")
    
    return found

def test_video_generation():
    """Test video generation with a short clip"""
    print("\n" + "="*50)
    print("Testing Video Generation")
    print("="*50)
    
    try:
        from video_generator import load_model, make_env
        
        print("\nLoading model...")
        model = load_model("mario_1_1_ppo.pt")
        print("✓ Model loaded")
        
        print("\nCreating environment...")
        env = make_env()
        print("✓ Environment created")
        
        print("\nRunning 1 episode to test...")
        obs = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Short test
        
        while steps < max_steps:
            import torch
            import numpy as np
            
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        env.close()
        
        print(f"✓ Test completed: {steps} steps, reward: {total_reward:.2f}")
        print("\n✓ Video generation system is ready!")
        print("\nTo generate videos, run:")
        print("  python3 video_generator.py --videos 1 --duration 2")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*50)
    print("Super Mario Video Generator Test")
    print("="*50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model files
    models_ok = check_model_files()
    
    if not deps_ok:
        print("\n" + "="*50)
        print("❌ Setup incomplete")
        print("="*50)
        print("\nPlease install dependencies:")
        print("  bash setup.sh")
        sys.exit(1)
    
    # Test video generation
    if not test_video_generation():
        print("\n" + "="*50)
        print("⚠️  Video generation test failed")
        print("="*50)
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)

if __name__ == "__main__":
    main()
