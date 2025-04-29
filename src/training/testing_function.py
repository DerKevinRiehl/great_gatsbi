"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This script contains implementations of functions to test a model.
"""




# #############################################################################
# IMPORTS
import torch
from models.model_loader import unpack_trajectory_prediction
from tqdm import tqdm




# #############################################################################
# METHODS

def test_model(model_name, model, test_loader, loss_functions, prediction_length, device, multimodal):
    model.eval()
    all_pred_trajs = []
    all_pred_trajs_a = []
    all_pred_trajs_b = []
    all_pred_trajs_c = []
    all_pred_trajs_d = []
    all_future_trajs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            batch_data = [x.to(device) for x in batch]
            future_traj = batch_data[0][:, :prediction_length, :]
            batch_feature_data = batch_data[1:]
            # Forward pass
            model_results = model(*batch_feature_data)
            model_res = unpack_trajectory_prediction(model_results, model_name, multimodal)
            if multimodal=="unimodal":
                pred_traj = model_res
                all_pred_trajs.append(pred_traj)
                all_future_trajs.append(future_traj)
            elif multimodal=="multimodal_gmm":
                mu_x = model_res[0]
                mu_y = model_res[1]
                sigma_x = model_res[2]
                sigma_y = model_res[3]
                rho = model_res[4]
                pi = model_res[5]
                # three evaluation ways
                pred_traj_a = best_mode(mu_x, mu_y, future_traj)
                pred_traj_b = most_probable_mode(mu_x, mu_y, pi)
                pred_traj_c = sampled_mode(mu_x, mu_y, sigma_x, sigma_y, rho, pi)       
                pred_traj_d = most_expected_mode(mu_x, mu_y, pi)
                all_pred_trajs_a.append(pred_traj_a)
                all_pred_trajs_b.append(pred_traj_b)
                all_pred_trajs_c.append(pred_traj_c)
                all_pred_trajs_d.append(pred_traj_d)
                all_future_trajs.append(future_traj)         
    
    # Concatenate all predictions and targets
    all_future_trajs = torch.cat(all_future_trajs, dim=0)
    if multimodal=="unimodal":
        all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    elif multimodal=="multimodal_gmm":
        all_pred_trajs_a = torch.cat(all_pred_trajs_a, dim=0)
        all_pred_trajs_b = torch.cat(all_pred_trajs_b, dim=0)
        all_pred_trajs_c = torch.cat(all_pred_trajs_c, dim=0)
        all_pred_trajs_d = torch.cat(all_pred_trajs_d, dim=0)
    
    # Evaluate model
    performances = {}
    for loss_function_name, loss_function in loss_functions.items():
        if multimodal=="unimodal":
            performances[loss_function_name] = loss_function(all_pred_trajs, all_future_trajs).item()
        elif multimodal=="multimodal_gmm":
            performances[loss_function_name] = [
                loss_function(all_pred_trajs_a, all_future_trajs).item(),
                loss_function(all_pred_trajs_b, all_future_trajs).item(),
                loss_function(all_pred_trajs_c, all_future_trajs).item(),
                loss_function(all_pred_trajs_d, all_future_trajs).item()
            ]
    return performances


def top_n_trajectories(mu_x, mu_y, pi, n=3):
    """
    Get the n most probable trajectories according to mixture weights.
    """
    batch_size, T_pred, num_modes = mu_x.shape

    # Find top-n modes per sample
    top_pi_values, top_pi_indices = pi.topk(n, dim=-1)  # [batch, T_pred, n]

    # We'll assume the modes are sorted same across timesteps (if pi is stable)

    # Select corresponding mu_x and mu_y
    top_mu_x = torch.gather(mu_x, dim=-1, index=top_pi_indices)  # [batch, T_pred, n]
    top_mu_y = torch.gather(mu_y, dim=-1, index=top_pi_indices)  # [batch, T_pred, n]

    return top_mu_x, top_mu_y, top_pi_values


def best_mode(mu_x, mu_y, gt):
    """
    Select best matching mode (lowest error) per example.
    Returns:
        mean_ade: float
        mean_fde: float
        best_traj: [batch, T_pred, 2] tensor of best mode trajectories
    """
    batch_size, T_pred, num_modes = mu_x.shape

    # Ground-truth
    gt_x = gt[..., 0].unsqueeze(-1)  # [batch, T_pred, 1]
    gt_y = gt[..., 1].unsqueeze(-1)  # [batch, T_pred, 1]

    # Compute L2 error per mode
    l2_error = torch.sqrt((mu_x - gt_x)**2 + (mu_y - gt_y)**2)  # [batch, T_pred, num_modes]

    # Sum over time for ADE
    ade_per_mode = l2_error.mean(dim=1)  # [batch, num_modes]

    # Select best mode per example
    best_ade, best_mode_idx = ade_per_mode.min(dim=1)  # [batch], [batch]

    # Collect best trajectories
    # Shape: [batch, T_pred, 2]
    best_mu_x = mu_x[torch.arange(batch_size).unsqueeze(1), torch.arange(T_pred), best_mode_idx.unsqueeze(1).expand(-1, T_pred)]
    best_mu_y = mu_y[torch.arange(batch_size).unsqueeze(1), torch.arange(T_pred), best_mode_idx.unsqueeze(1).expand(-1, T_pred)]
    best_traj = torch.stack([best_mu_x, best_mu_y], dim=-1)  # [batch, T_pred, 2]

    return best_traj


def most_probable_mode(mu_x, mu_y, pi): # final timestep
    """
    Select the most probable mode (highest pi at final timestep) per example.
    Args:
        mu_x: [batch, T_pred, num_modes]
        mu_y: [batch, T_pred, num_modes]
        pi:   [batch, T_pred, num_modes]
    Returns:
        most_prob_traj: [batch, T_pred, 2]
        most_prob_mode_idx: [batch]
    """
    batch_size, T_pred, num_modes = mu_x.shape

    # Choose mode with highest pi at final timestep
    most_prob_mode_idx = pi[:, -1, :].argmax(dim=1)  # [batch]

    # Prepare index tensor for gather
    idx = most_prob_mode_idx.unsqueeze(1).expand(-1, T_pred).unsqueeze(-1)  # [batch, T_pred, 1]

    # Gather along num_modes dimension
    most_prob_mu_x = torch.gather(mu_x, dim=2, index=idx).squeeze(2)  # [batch, T_pred]
    most_prob_mu_y = torch.gather(mu_y, dim=2, index=idx).squeeze(2)  # [batch, T_pred]

    most_prob_traj = torch.stack([most_prob_mu_x, most_prob_mu_y], dim=-1)  # [batch, T_pred, 2]

    return most_prob_traj


def most_expected_mode(mu_x, mu_y, pi):
    """
    Compute the expected trajectory as a linear combination of all modes weighted by their probabilities (pi).
    
    Args:
        mu_x: [batch, T_pred, num_modes] - x coordinates of the modes
        mu_y: [batch, T_pred, num_modes] - y coordinates of the modes
        pi:   [batch, T_pred, num_modes] - probability of each mode at each timestep
    
    Returns:
        expected_traj: [batch, T_pred, 2] - expected trajectory
    """
    batch_size, T_pred, num_modes = mu_x.shape
    
    # Compute expected x and y coordinates at each timestep
    expected_mu_x = torch.sum(pi * mu_x, dim=2)  # [batch, T_pred]
    expected_mu_y = torch.sum(pi * mu_y, dim=2)  # [batch, T_pred]
    
    # Stack to return the full trajectory
    expected_traj = torch.stack([expected_mu_x, expected_mu_y], dim=-1)  # [batch, T_pred, 2]
    
    return expected_traj


def sampled_mode(mu_x, mu_y, sigma_x, sigma_y, rho, pi):
    """
    Sample a single trajectory from the GMM.
    """
    batch_size, T_pred, num_modes = mu_x.shape
    device = mu_x.device

    # Sample a mode index according to pi
    pi = pi[:, 0, :]  # Use pi at first timestep, [batch, num_modes]
    sampled_mode = torch.multinomial(pi, num_samples=1)  # [batch, 1]

    # Gather parameters for the sampled mode
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    mu_x_sampled = mu_x[batch_indices, torch.arange(T_pred).unsqueeze(0), sampled_mode]  # [batch, T_pred]
    mu_y_sampled = mu_y[batch_indices, torch.arange(T_pred).unsqueeze(0), sampled_mode]
    sigma_x_sampled = sigma_x[batch_indices, torch.arange(T_pred).unsqueeze(0), sampled_mode]
    sigma_y_sampled = sigma_y[batch_indices, torch.arange(T_pred).unsqueeze(0), sampled_mode]
    rho_sampled = rho[batch_indices, torch.arange(T_pred).unsqueeze(0), sampled_mode]

    # Now sample from the 2D Gaussian for each timestep
    eps_x = torch.randn_like(mu_x_sampled)
    eps_y = torch.randn_like(mu_y_sampled)

    sampled_x = mu_x_sampled + sigma_x_sampled * eps_x
    sampled_y = mu_y_sampled + sigma_y_sampled * (rho_sampled * eps_x + torch.sqrt(1 - rho_sampled**2) * eps_y)

    sampled_traj = torch.stack([sampled_x, sampled_y], dim=-1)  # [batch, T_pred, 2]
    return sampled_traj

