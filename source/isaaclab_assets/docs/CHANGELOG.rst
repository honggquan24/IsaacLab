Changelog
---------

0.7.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Split the pendulum reward into two phases that meet at ``UPRIGHT_ANGLE`` (0.4 rad).
  ``swing_up_height`` rewards the height of the chain, ``mean(cos(error))``, and runs only while
  the chain is down; ``pendulum_is_upright`` and ``balance_pole_pos_l2`` run only once it is up. The
  swing-up phase uses a cosine rather than L2 because L2 reaches -π² ≈ -9.9 per step when the chain
  hangs, a constant large enough to drown out every other term.
* Added the bonus that makes the split safe. ``pendulum_is_upright`` is also used directly as a
  reward with weight 3.0, so crossing the threshold pays +2.85 against +1.83 just below it. Without
  it the swing-up term switches off at the boundary and leaves only the L2 penalty, which makes
  succeeding cost less than hovering just below the threshold forever.

Changed
^^^^^^^

* ``joint_pos_command_l2`` is now gated on the chain being upright, so a position command has no
  effect until the pendulum has been swung up. Chasing a marker while the pendulum hangs works
  directly against pumping energy, which needs the cart to sweep back and forth.
* The ``-Position`` tasks keep ``cart_pos`` at weight 0.05 instead of switching it off. While the
  chain is down the tracking term is gated off and ``cart_out_of_rail`` is a truncation carrying no
  penalty, so without it nothing would tell the cart to stay off the rail ends during swing-up.
* Carried the cart parameters set on the single pendulum over to the double and triple: 100 N effort
  limit, 20 m/s on the slider, 20 m/s on the bodies. The USD ``physxJoint:maxJointVelocity`` is
  regenerated to 20 m/s to match -- PhysX clamps to whatever is in the USD, so
  ``velocity_limit_sim`` alone would have had no effect.


0.6.1 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Sized the cart force to the rail instead of to the cart mass. Training ended 100% of episodes on
  ``cart_out_of_rail`` after a mean of 8.7 steps: at 40 N the 0.13 kg cart covers the 0.3 m from the
  reset spread to the rail limit in 2.6 control steps, while a quarter period of the 0.22 m pendulum
  is 11.5 steps, so the episode was over before the pendulum could move. The action scale drops to
  3 N (9.6 steps to the rail), the effort limit to 6 N and the slider ceiling to 2.5 m/s.
  Force was never the scarce resource: 3 N over a 0.5 m stroke is 1.5 J against the 0.024 J needed
  to raise the pendulum.
* Raised the ``cart_pos`` weight from 0.05 to 0.2. ``cart_out_of_rail`` is a truncation and carries
  no penalty of its own, so this term is the only signal telling the cart to stay off the rail ends.


0.6.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Rewrote the cart pendulum rewards in the shape Isaac Lab's own cartpole uses -- squared position
  error and absolute velocity, no exponentials. ``joint_pos_target_l2`` mirrors
  ``isaaclab_tasks.manager_based.classic.cartpole.mdp.rewards.joint_pos_target_l2``, wrapping the
  error to :math:`[-\pi, \pi]`, except that the target is ``default_joint_pos`` rather than a
  constant, since a chain has no single target angle. Velocity terms now call the core
  ``joint_vel_l1`` directly. The weights follow the shipped cartpole: alive 1.0, terminating -2.0,
  pole position -1.0, cart velocity -0.01, pole velocity -0.005.
* Removed ``upright_pendulum_exp`` and ``pendulum_upright_cos``. L2 already has a gradient at every
  angle, which is what swing-up needs, so the cosine shaping term is redundant.
* Marked ``cart_out_of_rail`` as ``time_out=True``. With an L2 penalty a hanging chain costs about
  :math:`-\pi^2` per step, so a failure termination worth -2.0 once would be the cheapest option
  available and the policy would learn to drive into the rail end on purpose. Flagging it as a
  truncation bootstraps the value and removes that incentive.
* Raised the cart authority again: action scale 15 to 40 N (about 300 m/s² on the single pendulum),
  effort limit 60 N, slider ceiling 12 to 20 m/s, drive damping 0.05 to 0.02.
* Doubled the control rate to 60 Hz (``decimation`` 2 to 1). At 30 Hz and 20 m/s the cart covers
  0.67 m per control step, over 60% of the rail, so raising the speed ceiling without raising the
  decision rate only buys collisions with the rail end. Every ``--video_length`` in the package
  docstrings doubles accordingly: 60 s is now 3600 steps.


0.5.1 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Gave the cart more authority on all three pendulums; at the previous settings it was too slow to
  respond. The action scale goes from 5 to 15 N (about 115 m/s² on the single pendulum, 58 m/s² on
  the triple) with the effort limit at 20 N, and the slider speed ceiling goes from 5 to 12 m/s in
  both the USD and ``velocity_limit_sim``.
* Cut the cart drive damping from 0.5 to 0.05. Damping is drag proportional to velocity and is
  subtracted from the control force, so at 5 m/s it was eating 2.5 N of a 5 N command.


0.5.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the ``cart_pendulum_triple`` package with ``Isaac-Cart-Pendulum-Triple``,
  ``-Triple-Position`` and ``-Triple-Position-Play``, and added the matching ``-Double-Position``
  tasks. The double and triple environment configs subclass the single-pendulum ones and only swap
  the robot, the viewer and the rail limit; every reward and termination resolves joints with
  ``Revolute_.*`` so a longer chain needs no new code.
* Added ``isaaclab_assets.cart_pendulum.mdp.events.reset_pendulum_chain``. Every revolute at zero is
  the chain hanging straight down and the default joint position is the chain standing up, so one
  ``hanging`` flag switches a task between swing-up and balance-only.
* Added ``pendulum_upright_cos``, a ``(1 + cos(error)) / 2`` shaping reward. ``upright_pendulum_exp``
  with ``std=0.35`` is about ``exp(-80)`` when the chain hangs, which is flat, so swing-up has
  nothing to follow without this term.

Changed
^^^^^^^

* Generalized ``scripts/ute/cart_pendulum/prepare_usd.py`` to any chain length. It discovers the
  articulation root, the bodies and the anchor joint from the stage and orients the joint tree by
  breadth-first search from the base body, so it no longer hard-codes prim names and works for the
  single, double and triple exports unchanged.
* ``prepare_usd.py`` now writes ``physxJoint:maxJointVelocity`` on every movable joint --
  ``--max-angular-velocity`` (15 rad/s) and ``--max-linear-velocity`` (5 m/s). A long chain reaches
  speeds within one simulation step that the solver cannot resolve, and capping the degree of
  freedom is far cheaper than shrinking ``sim.dt``. The actuator ``velocity_limit_sim`` values match.
* The cart pendulum tasks are now swing-up: the chain starts hanging and has to be swung up. The
  pendulum-fell termination is gone, since a fallen pendulum is the starting state, and the cart
  position penalty drops from 0.5 to 0.1 so it does not block the energy-pumping swings.
* The pendulum joints carry ``stiffness=1e-5`` with ``effort_limit_sim=1.0`` rather than a fully
  free joint, and the actor and critic grow to ``[512, 512, 256]``.
* Dropped ``articulation_root_prim_path`` from the robot configs. Left unset, Isaac Lab finds the
  prim carrying ``ArticulationRootAPI`` itself, so renaming the Onshape document no longer breaks
  the config.
* Reward and termination terms now pass ``asset_cfg`` explicitly, and ``resolve_joint_ids`` falls
  back to a name lookup, so a term that relies on the signature default no longer hits an unresolved
  ``slice(None)``.


0.4.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``scripts/ute/cart_pendulum/prepare_usd.py``, which turns the raw Onshape export
  ``cart_pendulum_base.usd`` into ``cart_pendulum_cfg.usd``: it reverses the parent/child order of
  ``Slider_1`` and ``Revolute_1``, replaces the D6 joint anchored to a helper plane with a
  ``UsdPhysics.FixedJoint`` to the world, lifts the assembly clear of the ground, and applies the
  missing ``UsdPhysics.DriveAPI``.
* Added ``isaaclab_assets.cart_pendulum.mdp.terminations`` with ``pendulum_fell`` and
  ``cart_out_of_rail``. ``pendulum_fell`` wraps the angle error to :math:`[-\pi, \pi]`, which
  ``joint_pos_out_of_manual_limit`` cannot do for a joint that spins without limits.

Changed
^^^^^^^

* Rebuilt the cart pendulum asset from a new Onshape import. The rail runs along **Y**, the slider
  limit is ±0.555 m, and joint zero is the pendulum hanging **down** — upright is :math:`\pi`, which
  is now the default joint position, so rewards measure deviation from the default rather than from
  zero.
* Rewrote ``isaaclab_assets.cart_pendulum.mdp.rewards``. The old ``cartpole_reward_joint_pos``
  compared pendulum velocity against a target of 2.0 rad/s, so it scored an upright, motionless
  pendulum near zero. The terms now resolve joints through :class:`SceneEntityCfg` and are split by
  what they measure.
* ``Isaac-Cart-Pendulum`` now ends an episode when the pendulum falls or the cart reaches the end of
  the rail, and the action drives only ``Slider_1`` (action dimension 2 to 1). The effort scale drops
  from 100 to 5 N to match the ~0.13 kg cart, and the viewer moved onto the X axis so the camera no
  longer looks down the length of the rail.
* ``CartPositionCommandCfg`` for this robot now uses ``rail_axis=(0, 1, 0)``.


0.3.1 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``isaaclab_assets.cart_pendulum.mdp.commands.CartPositionCommandCfg``, a command term that
  samples a target cart position along the rail. The sampling range follows the joint limits in the
  USD (``limit_ratio`` of the soft limits) so it does not need the rail length hard-coded, and its
  debug visualization draws the goal as a sphere on the rail.
* Added position-tracking reward terms to ``isaaclab_assets.cart_pendulum.mdp.rewards``:
  ``track_cart_position_exp``, ``cart_velocity_near_goal_l2``, ``upright_pendulum_exp`` and
  ``pendulum_ang_vel_l2``. These resolve joints through :class:`SceneEntityCfg` instead of hard-coded
  indices.
* Added the ``Isaac-Cart-Pendulum-Position`` task, where the cart balances the pendulum while driving
  to commanded positions, and ``Isaac-Cart-Pendulum-Position-Play``, a 60 s-episode variant for
  recording video.

Fixed
^^^^^

* Fixed the docstring of ``CartPendulumEnvCfg.TerminationsCfg``, which had unrelated text pasted into
  it.


0.3.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the HUTECH robot projects as packages under ``isaaclab_assets``: ``wheeled_biped``,
  ``rotary_pendulum``, ``cart_pendulum``, ``cart_pendulum_double``, ``balance_car`` and ``evobot``.
  Importing ``isaaclab_assets`` registers their Gymnasium tasks.
* Added ``isaaclab_assets.evobot.mdp.commands.BinaryGripperCommandCfg`` and
  ``isaaclab_assets.evobot.mdp.rewards_velocity`` so evoBOT no longer needs patches inside
  ``isaaclab.envs.mdp``.
* Added ``Isaac-Wheeled-Biped-Wheel-Play``, a low-env-count variant with a closer viewer for
  recording videos.

Changed
^^^^^^^

* Renamed the robot project packages to drop version suffixes. Update imports and task ids:
  ``legged_v5`` to ``wheeled_biped`` (``Isaac-Legged-V5-*`` to ``Isaac-Wheeled-Biped-*``),
  ``rotary_pendulum_v2`` to ``rotary_pendulum`` (``Isaac-RotaryPendulum-V2-*`` to
  ``Isaac-Rotary-Pendulum-*``), ``cartpole_v1`` to ``cart_pendulum`` (``Isaac-Cartpole-V1-Run``
  to ``Isaac-Cart-Pendulum``), ``cartpole_v2`` to ``cart_pendulum_double``
  (``Isaac-Cartpole-V2-Run`` to ``Isaac-Cart-Pendulum-Double``), ``balancecar_v1`` to
  ``balance_car`` (``Isaac-Cartbalance-V1-*`` to ``Isaac-Balance-Car-*``) and ``evobot_v1`` to
  ``evobot`` (``Isaac-Evobot-V1-*`` to ``Isaac-Evobot-*``). The ``experiment_name`` of every
  agent config is unchanged, so existing checkpoints under ``logs/rsl_rl`` still resolve.
* Changed each project to keep its USD files in ``usd/`` instead of ``usd_file/``.
* Changed the wheeled biped training scene to drop the follow camera, so training no longer
  requires ``--enable_cameras``.

Fixed
^^^^^

* Fixed :func:`isaaclab_assets.evobot.mdp.rewards_manipulation.undesired_contacts` so it selects
  ``sensor_cfg.body_ids`` and sums over bodies, instead of returning one value per body and
  breaking the reward manager.
* Fixed :func:`isaaclab_assets.evobot.mdp.rewards_manipulation.gripper_height_tracking_l2` to read
  the gripper height from ``body_pos_w`` relative to the root, instead of reading ``joint_pos``
  with a body-based ``SceneEntityCfg``.
* Fixed the evoBOT joint and body names to match ``evoBOT_v2_cfg.usd``: ``*_grabbing_joint``
  became ``*_gripper_joint`` and ``head_link`` became ``top_link``.
* Fixed ``EVOBOT_BALANCE_CFG`` to spawn ``evoBOT_v2_cfg.usd``. It pointed at a path that did not
  exist and, once resolved, at an edit layer with no contact reporting.

Removed
^^^^^^^

* Removed the ``legged_v1`` and ``legged_v2`` projects from this branch. They remain available on
  the ``legged_v3`` and ``dev/robot_legged_v2`` branches.
* Removed the follow camera from the wheeled biped scene. Any camera parented under the robot
  fails with ``TypeError: Unable to write from unknown dtype`` on Isaac Sim 5.1; record with
  ``play.py --video`` instead.

0.2.4 (2025-11-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Configuration for OpenArm robots used for manipulation tasks.

0.2.3 (2025-08-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Configuration for G1 robot used for locomanipulation tasks.

0.2.2 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Fourier GR1T2 robot.

0.2.1 (2025-01-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Humanoid-28 robot.


0.2.0 (2024-12-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Restructured the assets directory into ``robots`` and ``sensors`` subdirectories.


0.1.4 (2024-08-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Inverted Double Pendulum on a Cart robot.


0.1.2 (2024-04-03)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for different arms from Kinova Robotics and Rethink Robotics.


0.1.1 (2024-03-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for allegro and shadow hand assets.


0.1.0 (2023-12-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Moved all assets' configuration from ``isaaclab`` to ``isaaclab_assets`` extension.
