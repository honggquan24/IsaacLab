# Evobot V1 - Quick Start Guide

## Verify Installation

```bash
# Test all imports
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/test_imports.py

# List registered environments
./isaaclab.sh -p scripts/environments/list_envs.py | grep Evobot
```

## Training Commands

### 1. Balance Task (Recommended Starting Point)
```bash
# Production training (1024 parallel environments)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 1024 \
    --headless --rendering_mode performance

# Debug training (3 environments, with rendering)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 3 \
    --rendering_mode performance
```

### 2. Velocity Balance Task
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Velocity \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

### 3. Locomotion-Manipulation Task
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Locomotion-Manipulation \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

### 4. Hierarchical Navigation (Requires Pre-trained Balance Policy)

**Step 1**: Train balance task (see above)

**Step 2**: Export balance policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-Balance \
    --num_envs 1 \
    'agent.load_run=<balance_run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```
Policy saved to: `logs/rsl_rl/<algo>/<task>/<run_name>/exported/policy.pt`

**Step 3**: Update policy path
Edit: `source/isaaclab_assets/isaaclab_assets/evobot_v1/navigation/hierarchical/hierarchical_env_cfg.py`
Update: `policy_path = "path/to/exported/policy.pt"`

**Step 4**: Train navigation
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Navigation-Hierarchical \
    --num_envs 512 \
    --headless --rendering_mode performance
```

## Evaluation Commands

### Evaluate Trained Model
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-<TASK_NAME> \
    --num_envs 4 \
    'agent.load_run=<run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

### Hierarchical Navigation Evaluation
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Evobot-V1-Navigation-Hierarchical-Play \
    --num_envs 16 \
    'agent.load_run=<nav_run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

## Monitoring Training

```bash
# View TensorBoard logs
./isaaclab.sh -p -m tensorboard.main --logdir logs
```

Then open browser to: `http://localhost:6006`

## Directory Structure Reference

```
evobot_v1/
├── evobot_v1_cfg.py              # Robot configuration
├── balance/                       # Balance task
│   ├── evobot_v1_balance_env_cfg.py
│   ├── agents/rsl_rl_ppo_cfg.py
│   └── mdp/ (observations, rewards, terminations)
└── navigation/
    ├── shared_mdp/               # Shared navigation rewards
    ├── velocity/                 # Task 2: Velocity following
    │   ├── velocity_env_cfg.py
    │   ├── agents/rsl_rl_ppo_cfg.py
    │   └── manipulation_mdp.py
    ├── locomotion_manipulation/  # Task 3: Arm control
    │   ├── loc_man_env_cfg.py
    │   ├── agents/rsl_rl_ppo_cfg.py
    │   └── manipulation_mdp.py
    └── hierarchical/             # Task 4: Hierarchical nav
        ├── hierarchical_env_cfg.py
        ├── agents/rsl_rl_ppo_cfg.py
        └── mdp/ (actions, rewards)
```

## Troubleshooting

### Import Error: "No module named 'isaaclab_assets.evobot_v1...'"
**Solution**: Run import test to verify
```bash
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/test_imports.py
```

### Environment Not Found
**Solution**: Verify registration
```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep Isaac-Evobot
```

### Memory Issues During Training
**Solution**: Reduce number of environments or observation dimension
```bash
# Try with fewer envs
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 512  # reduced from 1024
```

### Policy Loading Error (Hierarchical)
**Solution**: Ensure policy path is correct in `hierarchical_env_cfg.py`
```python
# Check this line in hierarchical_env_cfg.py:
policy_path = "path/to/exported/policy.pt"  # Verify this exists!
```

## Key Hyperparameters

### Training Configuration
- **Learning Rate**: 1e-4
- **Num Steps Per Env**: 100
- **Max Iterations**: 400
- **Save Interval**: 20
- **Episode Length**: 50-100 steps (varies by task)

### Environment Configuration
- **Simulation Dt**: 0.01 seconds
- **Control Frequency**: 50 Hz
- **Gravity**: -9.81 m/s²

## Document References

| Document | Purpose |
|----------|---------|
| `EVOBOT.md` | Comprehensive robot & task documentation |
| `STRUCTURE.md` | Directory structure & organization |
| `EVOBOT_V1_REFACTORING_FINAL_REPORT.md` | Detailed refactoring report |
| `EVOBOT_QUICK_START.md` | This file - quick commands |

## Getting Help

1. **Check documentation**: `EVOBOT.md` in evobot_v1 directory
2. **Run tests**: Import test verifies all dependencies
3. **Isaac Lab docs**: https://isaac-sim.github.io/IsaacLab/
4. **GitHub issues**: Report bugs in repository

---

**Last Updated**: 2026-01-14
**Status**: All systems ready for training!

Quick test:
```bash
./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/evobot_v1/tests/test_imports.py
```
