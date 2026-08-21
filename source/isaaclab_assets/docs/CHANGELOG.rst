Changelog
---------

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
