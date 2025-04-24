"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This script contains implementations of the loss functions ADE (average displacement error) and FDE (final displacement error).
"""




# #############################################################################
# IMPORTS
import torch




# #############################################################################
# LOSS FUNCTIONS

def compute_ADE_train(pred, true):
    return torch.norm(pred - true, dim=2).mean()

def compute_FDE_train(pred, true):
    return torch.norm(pred[:, -1] - true[:, -1], dim=1).mean()

def compute_ADE(pred, true):
    # Compute L2 distance at each time step for each trajectory
    l2_dist = torch.norm(pred - true, dim=2)  # [N, T]
    ade = l2_dist.mean()  # Average over all points
    return ade.item()

def compute_FDE(pred, true):
    # Compute L2 distance at the final time step
    l2_dist_final = torch.norm(pred[:, -1] - true[:, -1], dim=1)  # [N]
    fde = l2_dist_final.mean()  # Average over all trajectories
    return fde.item()