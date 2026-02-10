# Rotary Pendulum V2 (Furuta Pendulum) - Configuration & Tasks Documentation

## Overview

**Rotary Pendulum V2** is a 2-DOF Furuta pendulum for reinforcement learning research. It features:
- **1 pivot motor** (Revolute_1 - actuated, rotates horizontal arm)
- **1 pendulum joint** (Revolute_2 - passive, free-swinging pendulum)

The control objective is to swing the pendulum from the hanging position to the inverted (upright) position and balance it there using only the pivot motor.

---

## Part 1: USD Robot Structure

### 1.1 Prim Hierarchy

```text
World (defaultPrim)
└── rotary_pendulum                        (Xform, namespace container)
    └── rotary_pendulum                    (Xform, ArticulationRoot)
        ├── pendulum                       (Xform) - Free-swinging pendulum arm
        ├── base                           (Xform) - Fixed base/mount
        └── pivot                          (Xform) - Rotating arm
            ├── Revolute_1                 (PhysicsRevolute) - Pivot motor joint
            └── Revolute_2                 (PhysicsRevolute) - Pendulum joint
```

### 1.2 Key Characteristics

- **Single ArticulationRoot**: All links/joints under one root
- **2 DOF**: 1 actuated (pivot motor) + 1 passive (pendulum)
- **Fixed base**: Base is mounted to the ground

### 1.3 Joint Definitions

| Joint | Type | Purpose | Actuated |
|-------|------|---------|----------|
| `Revolute_1` | Revolute | Pivot motor (horizontal rotation) | Yes |
| `Revolute_2` | Revolute | Pendulum (vertical swing) | No (passive) |

### 1.4 Link Definitions

| Link | Purpose |
|------|---------|
| `base` | Fixed base mount |
| `pivot` | Rotating horizontal arm |
| `pendulum` | Free-swinging vertical pendulum |

---

## Part 2: Isaac Lab Directory Structure

```
rotary_pendulum_v2/
├── rotary_pendulum_v2_cfg.py             # Robot configuration (root level)
├── __init__.py                           # Gym environment registration
├── .md                                   # This documentation file
│
├── mdp/                                  # MDP components
│   ├── __init__.py
│   ├── observations_balance.py           # Joint angles (sin/cos), velocities
│   ├── rewards_balance.py                # Pendulum upright, balance bonus, energy
│   └── terminations_balance.py           # Pivot angle limit
│
├── navigation/
│   ├── __init__.py
│   └── balance/                          # Swing-up / Balance task
│       ├── __init__.py
│       ├── rotary_pendulum_v2_balance_cfg.py
│       ├── rotary_pendulum_v2_balance_env_cfg.py
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py
│
└── usd_file/
    └── rotary_pendulum_v2_base.usd
```

---

## Part 3: Task Configuration

### Task: Swing-Up / Balance (Isaac-RotaryPendulum-V2-Balance)

**Objective**: Swing the pendulum from hanging position to inverted (upright) position and balance it.

**Observations (7D)**:
- `sin(theta1), cos(theta1)`: Pivot angle (2D)
- `sin(theta2), cos(theta2)`: Pendulum angle (2D)
- `dtheta1, dtheta2`: Joint velocities (2D)
- `last_action`: Previous pivot torque (1D)

**Actions (1D)**:
- Torque applied to pivot motor (Revolute_1)

**Rewards**:
1. `pendulum_upright` - Main reward: cos(theta2) approaching upright position
2. `balance_bonus` - Bonus when upright AND stable
3. `pendulum_vel_penalty` - Penalize high pendulum angular velocity
4. `pivot_vel_penalty` - Penalize excessive pivot spinning
5. `energy` - Minimize control effort
6. `action_rate` - Smooth actions
7. `alive` - Survival reward
8. `terminating` - Termination penalty

**Terminations**:
- Timeout after episode_length_s
- Pivot angle exceeds rotation limit

**Training**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RotaryPendulum-V2-Balance \
    --num_envs 4096 \
    --headless --rendering_mode performance
```

**Evaluation**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-RotaryPendulum-V2-Balance \
    --num_envs 4 \
    'agent.load_run=<run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

---

## Part 4: MDP Components

### Observations (in `observations_balance.py`)

| Function | Output | Description |
|----------|--------|-------------|
| `obs_joint_pos()` | (N, 2) | Raw joint positions [theta1, theta2] |
| `obs_joint_vel()` | (N, 2) | Joint velocities [dtheta1, dtheta2] |
| `obs_joint_pos_sin()` | (N, 2) | sin of angles [sin(theta1), sin(theta2)] |
| `obs_joint_pos_cos()` | (N, 2) | cos of angles [cos(theta1), cos(theta2)] |

### Rewards (in `rewards_balance.py`)

| Function | Description |
|----------|-------------|
| `pendulum_upright_reward()` | (1-cos(theta2))/2, max when upright |
| `balance_reward()` | Bonus when upright AND low velocity |
| `pendulum_angular_velocity_penalty()` | Penalize high dtheta2 |
| `pivot_velocity_penalty()` | Penalize high dtheta1 |
| `energy_penalty()` | Penalize applied torques |

### Terminations (in `terminations_balance.py`)

| Function | Description |
|----------|-------------|
| `reset_when_pivot_exceeds_limit()` | Terminate if pivot angle too large |

---

## Part 5: Gym Environment Registration

```python
gym.register(
    id="Isaac-RotaryPendulum-V2-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "...rotary_pendulum_v2_balance_env_cfg:RotaryPendulumV2BalanceEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:RotaryPendulumBalancePPORunnerCfg",
    },
)
```

**Available Tasks**:
- `Isaac-RotaryPendulum-V2-Balance` - Swing-up and balance task
