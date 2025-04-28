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
import math




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

def gmm_loss(mu_x, mu_y, sigma_x, sigma_y, rho, pi, gt):
    """
    mu_x, mu_y, sigma_x, sigma_y, rho, pi: [batch, T_pred, num_modes]
    gt: [batch, T_pred, 2] (ground truth)
    """
    x = gt[..., 0].unsqueeze(-1)  # [batch, T_pred, 1]
    y = gt[..., 1].unsqueeze(-1)

    norm_x = (x - mu_x) / sigma_x
    norm_y = (y - mu_y) / sigma_y

    z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    denom = 2 * (1 - rho**2)

    exponent = -z / denom
    normalizer = 2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2)

    component_prob = torch.exp(exponent) / normalizer  # [batch, T_pred, num_modes]

    weighted_prob = pi * component_prob  # Multiply by mixture weights

    prob = torch.sum(weighted_prob, dim=-1)  # Sum across modes

    nll = -torch.log(prob + 1e-10)  # Avoid log(0)
    return nll.mean()