# Evobot V1 - Robot Configuration & Tasks Documentation

## Overview

**Evobot V1** is a 5-DOF mobile manipulator robot designed for reinforcement learning research. It features:
- **2 wheels** (differential drive for locomotion)
- **1 arm** (articulated manipulator)
- **2 grippers** (grasping mechanism)
- **IMU sensor** (balance measurement)
- **Contact sensors** (force feedback)

This document covers both the USD (Universal Scene Description) structure and the Isaac Lab task implementations.

---

## Part 1: USD Robot Structure

### 1.1 Prim Hierarchy

```text
World (defaultPrim)
└── evobot                                  (Xform, namespace container)
    └── evobot                              (Xform, ArticulationRoot)
        │
        ├── top_link                        (Xform) - Main body (IMU mounted here)
        ├── head_link                       (Xform) - Head
        │   └── arm_link                    (Xform) - Arm
        │
        ├── leg_link                        (Xform) - Legs
        ├── wheel                           (Xform) - Right wheel
        ├── wheel_01                        (Xform) - Left wheel
        ├── gripper                         (Xform) - Left gripper
        ├── gripper_01                      (Xform) - Right gripper
        │
        ├── left_wheel_joint                (PhysicsRevoluteJoint)
        ├── right_wheel_joint               (PhysicsRevoluteJoint)
        ├── arm_joint                       (PhysicsRevoluteJoint)
        ├── left_grabbing_joint             (PhysicsPrismaticJoint)
        └── right_grabbing_joint            (PhysicsPrismaticJoint)
```

### 1.2 Key Characteristics

- **Single ArticulationRoot**: All links/joints under one root for stable physics solving
- **Flat joint hierarchy**: Joints are siblings of links (not nested)
- **5 DOF**: 2 wheels (rotation), 1 arm (rotation), 2 grippers (prismatic)

### 1.3 Joint Definitions

| Joint | Type | Purpose |
|-------|------|---------|
| `left_wheel_joint` | Revolute | Left wheel rotation |
| `right_wheel_joint` | Revolute | Right wheel rotation |
| `arm_joint` | Revolute | Arm rotation |
| `left_grabbing_joint` | Prismatic | Left gripper actuation |
| `right_grabbing_joint` | Prismatic | Right gripper actuation |

### 1.4 Link Definitions

| Link | Purpose |
|------|---------|
| `top_link` | Main body / root link (IMU mounted) |
| `head_link` | Head assembly |
| `arm_link` | Arm assembly |
| `leg_link` | Leg assembly |
| `wheel`, `wheel_01` | Wheels (right/left) |
| `gripper`, `gripper_01` | Grippers (right/left) |

---

## Part 2: Isaac Lab Directory Structure

### 2.1 New Optimized Structure

```
evobot_v1/
├── evobot_v1_cfg.py                      # ★ Robot configuration (root level)
│
├── balance/                              # ★ BALANCE TASK
│   ├── __init__.py
│   ├── evobot_v1_balance_env_cfg.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rsl_rl_ppo_cfg.py
│   └── mdp/                              # Task-specific MDP terms
│       ├── __init__.py
│       ├── observations.py               # IMU, joint, pose observations
│       ├── rewards.py                    # 7 balance reward functions
│       └── terminations.py               # Fall detection
│
├── mdp/                                  # ★ CONSOLIDATED MDP COMPONENTS
│   ├── __init__.py                       # Exports all MDP functions
│   ├── observations_balance.py           # Balance observations
│   ├── rewards_balance.py                # Balance rewards
│   ├── terminations_balance.py           # Balance terminations
│   ├── rewards_navigation.py             # Navigation shared rewards
│   ├── rewards_hierarchical.py           # Hierarchical navigation rewards
│   ├── rewards_manipulation.py           # Manipulation utilities
│   ├── actions_hierarchical.py           # Pre-trained policy wrapper
│   └── (other mdp files)
│
├── navigation/                           # ★ NAVIGATION TASKS (3 variants)
│   ├── __init__.py
│   │
│   ├── velocity/                         # Task 1: Velocity Balance
│   │   ├── __init__.py
│   │   ├── velocity_env_cfg.py
│   │   └── agents/
│   │       ├── __init__.py
│   │       └── rsl_rl_ppo_cfg.py         # EvobotVelocityPPORunnerCfg
│   │
│   ├── locomotion_manipulation/          # Task 2: Manipulation
│   │   ├── __init__.py
│   │   ├── loc_man_env_cfg.py
│   │   └── agents/
│   │       ├── __init__.py
│   │       └── rsl_rl_ppo_cfg.py         # 
│   │
│   └── hierarchical/                     # Task 3: Hierarchical Navigation
│       ├── __init__.py
│       ├── hierarchical_env_cfg.py
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py         # EvobotNavigationPPORunnerCfg
│
├── tests/
│   └── run_robot_rl_env.py
├── usd_file/
│   └── evoBOT_cfg.usd
│
├── __init__.py                           # Gym environment registration
└── EVOBOT.md                             # This file
```

### 2.2 Why This Structure?

1. **Robot config at root**: Used by all tasks → placed at root level
2. **Consolidated MDP at root level**: All MDP components (observations, rewards, terminations, actions) in single `mdp/` directory
   - `observations_balance.py` - Balance task observations
   - `rewards_balance.py`, `rewards_navigation.py`, `rewards_hierarchical.py`, `rewards_manipulation.py` - Task-specific rewards
   - `terminations_balance.py` - Balance task terminations
   - `actions_hierarchical.py` - Pre-trained policy action wrapper
3. **Balance task self-contained**: All balance files in `balance/` directory (uses root-level mdp)
4. **Navigation tasks organized by variant**:
   - `velocity/` - Simple velocity tracking
   - `locomotion_manipulation/` - With arm control
   - `hierarchical/` - Uses pre-trained balance policy
   - Each has own config and agent files (uses root-level mdp)
5. **No duplication**: Each function appears in exactly one place
6. **Clean imports**: All tasks import from `..mdp` (parent mdp directory)

---

## Part 3: Task Configurations

### Task 1: Balance (Isaac-Evobot-V1-Balance)

**Objective**: Maintain upright standing position (self-balancing robot)

**Key Features**:
- **Observations**: IMU (accel, gyro, orientation, gravity), joint states, pose
- **Actions**: 5D joint effort (2 wheels + arm + 2 grippers)
- **Rewards**:
  - RPY alignment (upright orientation)
  - Low angular velocity (stability)
  - Low linear velocity (standing still)
  - Survival bonus
  - Contact force symmetry
- **Terminations**: Falls, too low, bad orientation, excessive contact

**Training**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Balance \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

**Files**:
- Config: `balance/evobot_v1_balance_env_cfg.py`
- MDP: `balance/mdp/{observations,rewards,terminations}.py`
- PPO: `balance/agents/rsl_rl_ppo_cfg.py`

---

### Task 2: Velocity Balance (Isaac-Evobot-V1-Velocity)

**Objective**: Balance while following velocity commands

**Key Features**:
- Extends balance task with velocity command following
- **New observations**: Velocity commands
- **New rewards**: Position and heading tracking

**Training**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Velocity \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

**Files**:
- Config: `navigation/velocity/velocity_env_cfg.py`
- PPO: `navigation/velocity/agents/rsl_rl_ppo_cfg.py` (EvobotVelocityPPORunnerCfg)
- Shared rewards: `mdp/rewards_navigation.py`

---

### Task 3: Locomotion-Manipulation (Isaac-Evobot-V1-Locomotion-Manipulation)

**Objective**: Balance, navigate, and control arm simultaneously

**Key Features**:
- Combines balance + velocity tracking + arm control
- All 5 DOF active (wheels + arm + grippers)
- **New observations**: Arm/gripper states
- **New rewards**: Manipulation success metrics

**Training**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Locomotion-Manipulation \
    --num_envs 1024 \
    --headless --rendering_mode performance
```

**Files**:
- Config: `navigation/locomotion_manipulation/loc_man_env_cfg.py`
- PPO: `navigation/locomotion_manipulation/agents/rsl_rl_ppo_cfg.py` (EvobotLocomotionManipulationPPORunnerCfg)
- Rewards: `mdp/rewards_manipulation.py`, `mdp/rewards_navigation.py`

---

### Task 4: Hierarchical Navigation (Isaac-Evobot-V1-Navigation-Hierarchical)

**Objective**: Navigate to target using pre-trained balance policy as low-level controller

**Architecture**:
```
High-level Policy (navigation)
    ↓ outputs [vx, vy, omega]
PreTrainedBalancePolicyAction wrapper
    ↓ runs balance policy on low-level observations
Low-level Policy (balance checkpoint)
    ↓ outputs wheel torques
Robot wheels
```

**Workflow**:
1. Train balance task first: `--task=Isaac-Evobot-V1-Balance`
2. Export policy checkpoint
3. Update `policy_path` in `hierarchical/hierarchical_env_cfg.py`
4. Train navigation: `--task=Isaac-Evobot-V1-Navigation-Hierarchical`

**Training Navigation**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Evobot-V1-Navigation-Hierarchical \
    --num_envs 512 \
    --headless --rendering_mode performance
```

**Evaluation**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Evobot-V1-Navigation-Hierarchical-Play \
    --num_envs 16 \
    'agent.load_run=<run_name>' \
    'agent.load_checkpoint="model_500.pt"'
```

**Files**:
- Config: `navigation/hierarchical/hierarchical_env_cfg.py`
- Action wrapper: `mdp/actions_hierarchical.py`
- Rewards: `mdp/rewards_hierarchical.py`, `mdp/rewards_navigation.py`
- PPO: `navigation/hierarchical/agents/rsl_rl_ppo_cfg.py` (EvobotNavigationPPORunnerCfg)

---

## Part 4: MDP Components

### All MDP Components (Consolidated)

**Location**: `mdp/` directory (root level)

**Observations** (in `observations_balance.py`):
- `obs_body_roll/pitch/yaw()` - Euler angles from IMU
- `lin_vel_b()` - Body frame linear velocity
- `angl_vel_b()` - Body frame angular velocity
- `obs_pos_world()` - Position relative to env origin

**Balance Rewards** (in `rewards_balance.py`):
1. `rpy_alignment_imu()` - Main balance reward
2. `angular_velocity_reward()` - Stability
3. `linear_velocity_reward()` - Standing still
4. `height_reward()` - Maintain height
5. `joint_pos_target_l2()` - Joint position tracking
6. `joint_force_balance()` - Symmetric forces
7. `feet_contact_force_symmetry()` - Contact balance

**Navigation Rewards** (in `rewards_navigation.py`):
- Position tracking: `position_command_error_tanh()`, `position_reached_bonus()`
- Heading: `heading_command_error_abs()`, `heading_alignment_reward()`
- Velocity: `navigation_velocity_reward()`, `forward_velocity_tracking()`
- Progress: `goal_progress_reward()`, `velocity_goal_alignment()`
- Stability: `upright_reward()`, `tilt_penalty()`, `yaw_rate_penalty()`
- Penalties: `joint_velocity_penalty()`, `lateral_velocity_penalty()`

**Hierarchical Rewards** (in `rewards_hierarchical.py`):
- Same as navigation with alternative implementations

**Manipulation Utilities** (in `rewards_manipulation.py`):
- `reward_wheel_speed()`, `action_rate_l2()`, `joint_acc_l2()`
- `undesired_contacts()`, `reward_man()`

**Terminations** (in `terminations_balance.py`):
- `reset_when_fall()` - Excessive tilt angle

**Actions** (in `actions_hierarchical.py`):
- `PreTrainedBalancePolicyAction` - Wrapper for pre-trained balance policy

---

## Part 5: Gym Environment Registration

Environments are registered in `evobot_v1/__init__.py`:

```python
gym.register(
    id="Isaac-Evobot-V1-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_assets.evobot_v1.balance.evobot_v1_balance_env_cfg:EvobotV1BalanceEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_assets.evobot_v1.balance.agents.rsl_rl_ppo_cfg:EvobotBalancePPORunnerCfg"
    },
)
```

**Available Tasks**:
- `Isaac-Evobot-V1-Balance`
- `Isaac-Evobot-V1-Velocity`
- `Isaac-Evobot-V1-Locomotion-Manipulation`
- `Isaac-Evobot-V1-Navigation-Hierarchical`
- `Isaac-Evobot-V1-Navigation-Hierarchical-Play`

---