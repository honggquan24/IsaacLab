#!/usr/bin/env python3
"""Quick verification script for PID extraction logic.

Tests the mathematical decomposition:
    left_wheel = pitch + roll
    right_wheel = pitch - roll

Should recover:
    pitch = (left + right) / 2
    roll = (left - right) / 2
"""

import numpy as np

def test_pid_decomposition():
    """Test PID gain decomposition logic."""

    print("\n" + "="*80)
    print("TESTING PID DECOMPOSITION LOGIC")
    print("="*80)

    # Ground truth PID gains
    kp_roll_true = 1.5
    ki_roll_true = 0.05
    kd_roll_true = 0.4

    kp_pitch_true = 2.0
    ki_pitch_true = 0.1
    kd_pitch_true = 0.6

    print("\nGround Truth PID Gains:")
    print(f"  Roll:  Kp={kp_roll_true}, Ki={ki_roll_true}, Kd={kd_roll_true}")
    print(f"  Pitch: Kp={kp_pitch_true}, Ki={ki_pitch_true}, Kd={kd_pitch_true}")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Random errors and their derivatives/integrals
    roll_error = np.random.randn(n_samples) * 0.2
    roll_integral = np.cumsum(roll_error) * 0.01
    roll_vel = np.random.randn(n_samples) * 0.5

    pitch_error = np.random.randn(n_samples) * 0.2
    pitch_integral = np.cumsum(pitch_error) * 0.01
    pitch_vel = np.random.randn(n_samples) * 0.5

    # Compute control outputs using true PID
    roll_control = (
        kp_roll_true * roll_error +
        ki_roll_true * roll_integral +
        kd_roll_true * roll_vel
    )

    pitch_control = (
        kp_pitch_true * pitch_error +
        ki_pitch_true * pitch_integral +
        kd_pitch_true * pitch_vel
    )

    # Wheel mapping
    left_wheel = pitch_control + roll_control
    right_wheel = pitch_control - roll_control

    print(f"\nGenerated {n_samples} samples")
    print(f"  Left wheel range:  [{left_wheel.min():.3f}, {left_wheel.max():.3f}]")
    print(f"  Right wheel range: [{right_wheel.min():.3f}, {right_wheel.max():.3f}]")

    # Prepare feature matrix for regression
    X = np.column_stack([
        roll_error,
        roll_integral,
        roll_vel,
        pitch_error,
        pitch_integral,
        pitch_vel,
    ])

    # Perform linear regression (solve normal equations)
    # Left wheel = X @ coef_left
    coef_left = np.linalg.lstsq(X, left_wheel, rcond=None)[0]
    coef_right = np.linalg.lstsq(X, right_wheel, rcond=None)[0]

    print("\nLinear Regression Coefficients:")
    print(f"  Left:  {coef_left}")
    print(f"  Right: {coef_right}")

    # Decompose to recover roll and pitch gains
    # pitch = (left + right) / 2
    # roll = (left - right) / 2

    roll_gains_recovered = (coef_left[:3] - coef_right[:3]) / 2
    pitch_gains_recovered = (coef_left[3:] + coef_right[3:]) / 2

    kp_roll_recovered = roll_gains_recovered[0]
    ki_roll_recovered = roll_gains_recovered[1]
    kd_roll_recovered = roll_gains_recovered[2]

    kp_pitch_recovered = pitch_gains_recovered[0]
    ki_pitch_recovered = pitch_gains_recovered[1]
    kd_pitch_recovered = pitch_gains_recovered[2]

    print("\nRecovered PID Gains:")
    print(f"  Roll:  Kp={kp_roll_recovered:.4f}, Ki={ki_roll_recovered:.4f}, Kd={kd_roll_recovered:.4f}")
    print(f"  Pitch: Kp={kp_pitch_recovered:.4f}, Ki={ki_pitch_recovered:.4f}, Kd={kd_pitch_recovered:.4f}")

    # Compute errors
    roll_errors = [
        abs(kp_roll_recovered - kp_roll_true),
        abs(ki_roll_recovered - ki_roll_true),
        abs(kd_roll_recovered - kd_roll_true),
    ]

    pitch_errors = [
        abs(kp_pitch_recovered - kp_pitch_true),
        abs(ki_pitch_recovered - ki_pitch_true),
        abs(kd_pitch_recovered - kd_pitch_true),
    ]

    print("\nRecovery Errors (should be ~0):")
    print(f"  Roll:  ΔKp={roll_errors[0]:.6f}, ΔKi={roll_errors[1]:.6f}, ΔKd={roll_errors[2]:.6f}")
    print(f"  Pitch: ΔKp={pitch_errors[0]:.6f}, ΔKi={pitch_errors[1]:.6f}, ΔKd={pitch_errors[2]:.6f}")

    # Check if recovery is successful
    max_error = max(max(roll_errors), max(pitch_errors))

    print("\n" + "="*80)
    if max_error < 1e-10:
        print("✓ TEST PASSED: PID decomposition logic is CORRECT!")
        print(f"  Maximum recovery error: {max_error:.2e}")
    else:
        print("✗ TEST FAILED: PID decomposition has errors")
        print(f"  Maximum recovery error: {max_error:.2e}")
    print("="*80 + "\n")

    return max_error < 1e-10


if __name__ == "__main__":
    success = test_pid_decomposition()
    exit(0 if success else 1)
