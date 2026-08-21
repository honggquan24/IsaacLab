# Curriculum Learning Guide for Rotary Pendulum V2

## 📚 Overview

Curriculum learning breaks the complex swing-up + balance + heading tracking task into 2 progressive stages:

1. **Stage 1: Swing-Up Only** - Learn basic skill (get pendulum upright)
2. **Stage 2: Heading Tracking** - Add complexity (track pivot angle while balanced)

This approach significantly improves learning efficiency and final performance.

---

## 🎯 Stage Breakdown

### **Stage 1: Swing-Up Only** (`Isaac-RotaryPendulum-V2-Balance-Stage1`)

**Goal**: Get pendulum from hanging (0°) to upright (180°)

**Characteristics**:
- ✅ **Simpler observations** (7D): No heading command
- ✅ **Longer episodes** (20s): More time for exploration
- ✅ **Larger pivot range** (±180°): Freedom to swing
- ✅ **Random initial conditions**: Better generalization
- ✅ **Simpler reward structure**: Focus on swing-up
- ✅ **Higher exploration noise**: Discover swing-up strategy

**Key Rewards**:
```python
pendulum_upright:   weight=5.0   # cos-based: 0 (hanging) → 1 (upright)
balance_bonus:      weight=10.0  # Extra reward when stable
energy:             weight=0.05  # Encourage swing, penalize when upright
```

**Termination**:
- Pivot exceeds ±180° (full range)
- Episode timeout: 20 seconds

---

### **Stage 2: Balance + Heading Tracking** (`Isaac-RotaryPendulum-V2-Balance-Stage2`)

**Goal**: Maintain balance while tracking commanded pivot heading

**Characteristics**:
- ✅ **Full observations** (10D): Includes heading command
- ✅ **Tighter termination** (±90°): Controlled tracking
- ✅ **Shorter episodes** (15s): Tracking task
- ✅ **Load Stage 1 checkpoint**: Warm start
- ✅ **Lower exploration noise**: Fine-tuning
- ✅ **Added reward**: Pivot heading tracking

**Additional Rewards**:
```python
pivot_heading_tracking: weight=-2.0  # Only active when balanced
balance_bonus:          weight=20.0  # Higher emphasis on stability
```

**Termination**:
- Pivot exceeds ±90° (match heading range)
- Episode timeout: 15 seconds

---

## 🚀 Training Workflow

### **Step 1: Train Stage 1 (Swing-Up)**

```bash
# Production training (4096 envs, headless)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RotaryPendulum-V2-Balance-Stage1 \
    --num_envs 4096 \
    --headless \
    --rendering_mode performance

# Debug training (small num_envs, with rendering)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RotaryPendulum-V2-Balance-Stage1 \
    --num_envs 16 \
    --rendering_mode performance
```

**Training Hyperparameters**:
- **Max iterations**: 1000
- **Steps per env**: 1200 (20s episodes)
- **Exploration noise**: 1.0 (high)
- **Entropy coefficient**: 0.02 (high)
- **Learning rate**: 1e-3

**What to Monitor**:
- `episode_rew_mean`: Should increase from ~-50 to ~5+
- `episode_len_mean`: Should stay close to 1200 (full episodes)
- TensorBoard: Watch `pendulum_upright` and `balance_bonus` rewards

**Success Criteria**:
- ✅ `episode_rew_mean` > 5.0
- ✅ Agent swings up and balances consistently (watch visual)
- ✅ Save checkpoint from best iteration (e.g., `model_800.pt`)

---

### **Step 2: Train Stage 2 (Heading Tracking)**

**IMPORTANT**: Load the best checkpoint from Stage 1!

```bash
# Resume from Stage 1 checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RotaryPendulum-V2-Balance-Stage2 \
    --num_envs 4096 \
    --resume \
    --load_run=<YYYY-MM-DD_HH-MM-SS> \
    --checkpoint=model_800.pt \
    --headless \
    --rendering_mode performance
```

**Example**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RotaryPendulum-V2-Balance-Stage2 \
    --num_envs 4096 \
    --resume \
    --load_run=2025-02-09_14-30-45 \
    --checkpoint=model_800.pt \
    --headless
```

**Training Hyperparameters**:
- **Max iterations**: 500 (fine-tuning)
- **Steps per env**: 900 (15s episodes)
- **Exploration noise**: 0.3 (lower)
- **Entropy coefficient**: 0.005 (lower)
- **Learning rate**: 5e-4 (lower)

**What to Monitor**:
- `episode_rew_mean`: Should stay high (agent already knows swing-up)
- `pivot_heading_tracking`: Should decrease (lower error)
- Visual: Watch pendulum track different heading commands

**Success Criteria**:
- ✅ Maintains swing-up performance
- ✅ Successfully tracks heading commands while balanced
- ✅ Low heading tracking error

---

## 📊 Evaluation

### **Evaluate Stage 1**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-RotaryPendulum-V2-Balance-Stage1 \
    --num_envs 4 \
    'agent.load_run=2025-02-09_14-30-45' \
    'agent.load_checkpoint="model_800.pt"'
```

### **Evaluate Stage 2**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-RotaryPendulum-V2-Balance-Stage2 \
    --num_envs 4 \
    'agent.load_run=2025-02-09_16-45-20' \
    'agent.load_checkpoint="model_400.pt"'
```

---

## 📈 Expected Learning Curves

### **Stage 1: Swing-Up**

```
Iteration    episode_rew_mean    Comments
---------    ----------------    --------
0-100        -80 to -50          Random exploration
100-300      -50 to -10          Discovering swing motion
300-600      -10 to 0            Reaching upright occasionally
600-1000     0 to 10             Consistent swing-up + balance
```

### **Stage 2: Heading Tracking**

```
Iteration    episode_rew_mean    Comments
---------    ----------------    --------
0-50         5 to 8              Adapting to heading commands
50-200       8 to 12             Learning to track
200-500      12 to 15            Fine-tuning tracking accuracy
```

---

## 🔧 Troubleshooting

### **Stage 1 not learning swing-up**

**Symptoms**: Reward stuck around -80, pendulum stays hanging

**Solutions**:
1. ✅ Increase exploration: `init_noise_std=1.5` in PPO config
2. ✅ Check termination: Should be ±180° (full range)
3. ✅ Verify random reset: `position_range=(-0.2, 0.2)`
4. ✅ Increase episode length: 30s if needed

### **Stage 2 forgets swing-up**

**Symptoms**: Agent can't swing up anymore after loading checkpoint

**Solutions**:
1. ✅ **Check checkpoint loading**: Verify run name and model number
2. ✅ Lower learning rate: Try `1e-4` instead of `5e-4`
3. ✅ Increase `clip_param`: From 0.2 to 0.3 to preserve old policy
4. ✅ Start with higher weight on swing-up rewards initially

### **Stage 2 can't track heading**

**Symptoms**: Swings up but ignores heading command

**Solutions**:
1. ✅ Increase `pivot_heading_tracking` weight: From -2.0 to -5.0
2. ✅ Decrease other reward weights to make heading more salient
3. ✅ Verify observation includes heading: Should be 10D
4. ✅ Check command resampling: 5-10s should provide varied targets

---

## 🎓 Key Differences from Original Config

| Aspect | Original | Stage 1 | Stage 2 |
|--------|----------|---------|---------|
| **Observations** | 10D (with heading) | **7D** (no heading) | 10D (with heading) |
| **Episode Length** | 10s | **20s** | 15s |
| **Pivot Range** | ±90° | **±180°** | ±90° |
| **Initial Noise** | 0.5 | **1.0** | 0.3 |
| **Entropy Coef** | 0.01 | **0.02** | 0.005 |
| **Learning Rate** | 1e-3 | 1e-3 | **5e-4** |
| **Heading Tracking** | ✅ Active | ❌ Disabled | ✅ Active |
| **Random Reset** | ❌ Disabled | ✅ **Enabled** | ✅ Enabled |
| **Max Iterations** | 500 | **1000** | 500 |

---

## 💡 Tips for Success

1. **Monitor TensorBoard**:
   ```bash
   ./isaaclab.sh -p -m tensorboard.main --logdir logs
   ```

2. **Save checkpoints regularly**: Use `save_interval=100` for Stage 1

3. **Test intermediate checkpoints**: Don't wait until iteration 1000, test at 500, 700, etc.

4. **Visual verification**: Always watch a few episodes to verify behavior

5. **Hyperparameter tuning**: If Stage 1 doesn't work after 500 iterations, adjust exploration

6. **Stage 2 can be shorter**: If agent already tracks well after 200 iterations, stop early

7. **Experiment with reward weights**: Balance between swing-up and tracking based on your needs

---

## 📁 Files Structure

```
rotary_pendulum_v2/
├── navigation/balance/
│   ├── rotary_pendulum_v2_balance_env_cfg.py      # Original config
│   ├── rotary_pendulum_v2_balance_curriculum.py   # ⭐ NEW: Curriculum configs
│   ├── agents/
│   │   ├── rsl_rl_ppo_cfg.py                      # Original PPO
│   │   └── rsl_rl_ppo_curriculum_cfg.py           # ⭐ NEW: Curriculum PPO
│   └── __init__.py
├── __init__.py                                     # Gym registration
└── CURRICULUM_LEARNING_GUIDE.md                   # ⭐ This file
```

---

## 🎯 Summary

**Curriculum learning significantly improves sample efficiency and final performance!**

- ✅ Stage 1: Master swing-up (harder to discover)
- ✅ Stage 2: Add heading tracking (easier when already balanced)
- ✅ Better final policy than training everything at once
- ✅ Faster convergence

**Next Steps**: Run Stage 1 training and monitor progress! 🚀
