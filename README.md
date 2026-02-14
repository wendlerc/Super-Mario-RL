# :mushroom: Super-Mario-RL 

*Tested by Flux 🌊 - SSH access confirmed!*

This is a private project to make Super Mario Agent.

It consists of training an agent to clear Super Mario Bros with deep reinforcement learning methods.

Here are my super mario agents with dueling network. ( trained 7,000 epoch )

**(25-05-20) SuperMario with PPO has been updated!**

<p float="center">
  <img src="/mario1.gif" width="350" />
  <img src="/mario14.gif" width="350" /> 
</p>

# Get started

## Cloning git

```
git clone https://github.com/jiseongHAN/Super-Mario-RL.git
cd Super-Mario-RL
```

## Install Requirements
```
pip install -r requirements.txt
```

## Or Install Manually
* Install [openAI gym](http://gym.openai.com/)
```
pip install 'gym'
```
* Install [Pytorch](https://pytorch.org/)
```
pip install torch torchvision
```
* Install [nes-py](https://pypi.org/project/nes-py/)
```
pip install nes-py
```
* Install [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/)
```
pip install gym-super-mario-bros
```

# Running

## Train

* Train with dueling dqn.
```
python duel_dqn.py
```

* Train with PPO.

```
python ppo.py
```

### Result
* score.p : save total score every 50 episode
* *.pth : save weight of q, q_target every 50 training


## Evaluate
* (Now, pre-trained agent has been corrupted😢)
* Test and render trained agent.
* To test our agent, we need 'q_target.pth' that generated at the training step.
* (eval.py with PPO is not supported now)
```
python eval.py
```
* Or you can use your own agent.
```
python eval.py your_own_agent.pth
```

## Video Dataset Generation

Generate gameplay videos with annotated actions for training datasets.

### Quick Start

1. **Setup environment:**
```bash
bash setup.sh
```

2. **Test the setup:**
```bash
python3 test_setup.py
```

3. **Generate a single video (2 minutes):**
```bash
python3 video_generator.py --videos 1 --duration 2
```

### Generate multiple videos for dataset
```bash
python3 video_generator.py --videos 10 --duration 5
```

### Video annotations include:
- **Action display**: Current button presses (UP, DOWN, LEFT, RIGHT, A, B)
- **Reward tracking**: Real-time reward values
- **Game state**: Stage, score, coins, lives
- **Position**: X-coordinate progress
- **Visual indicators**: Button press visualization

Videos are saved to `videos/` directory with timestamps.

### Requirements
```
pip install -r requirements.txt
```

### Example output
```
Recording 2 minutes of gameplay...
Output: videos/mario_gameplay_20250214_203456.mp4
Progress: 50.0% - Reward: 1250.50 - Stage: 1

Video saved to: videos/mario_gameplay_20250214_203456.mp4
Total frames: 3600
Total reward: 2847.32
Duration: 120.0 seconds
```

## Reference
[Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://arxiv.org/pdf/1511.06581.pdf)

[Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal policy optimization
algorithms. arXiv preprint arXiv:1707.06347 (2017).](https://arxiv.org/pdf/1707.06347)
