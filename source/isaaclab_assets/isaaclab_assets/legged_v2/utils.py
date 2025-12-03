# import torch
# from isaaclab_assets import Articulation
# from isaaclab_assets import SceneEntityCfg

# def custom_joint_pos_limit(
#     env, 
#     joint_limits: dict = {}, ):
#     robot = env.scene['robot']
#     joint_pos = robot.data.joint_pos  # shape: [num_envs, num_joints]
#     terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
#     if joint_limits:
        