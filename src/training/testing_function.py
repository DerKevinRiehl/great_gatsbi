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

def test_model(model_name, model, test_loader, loss_functions, prediction_length, device):
    model.eval()
    all_pred_trajs = []
    all_future_trajs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            batch_data = [x.to(device) for x in batch]
            future_traj = batch_data[0][:, :prediction_length, :]
            batch_feature_data = batch_data[1:]
            # Forward pass
            model_results = model(*batch_feature_data)
            pred_traj = unpack_trajectory_prediction(model_results, model_name)
            # print(len(pred_traj[0]), len(future_traj[0]))
            all_pred_trajs.append(pred_traj)
            all_future_trajs.append(future_traj)
    
    # Concatenate all predictions and targets
    all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    all_future_trajs = torch.cat(all_future_trajs, dim=0)
    
    # Evaluate model
    performances = {}
    for loss_function_name, loss_function in loss_functions.items():
        performances[loss_function_name] = loss_function(all_pred_trajs, all_future_trajs).item()
    
    return performances
