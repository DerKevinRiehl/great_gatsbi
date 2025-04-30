"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains utils for models.
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F




# #############################################################################
# ### METHODS

def output_layer_unimodal(hidden_dim, output_dim):
    # (x,y)
    # return nn.Linear(hidden_dim, output_dim)
    return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

def output_layer_multimodal_gmm(hidden_dim, num_modes):
    # Instead of 2 outputs (x, y), we predict for each mode:
    # (mu_x, mu_y, sigma_x, sigma_y, correlation rho) + mixture weight
    return nn.Linear(hidden_dim, num_modes * 6)  # 6 params per mode

def output_decode_unimodal(h_dec, output_model):
    return output_model(h_dec)
        
def output_decode_multimodal_gmm(h_dec, num_modes, output_model):
    # --- output head (CHANGED) ---
    raw_output = output_model(h_dec)  # [batch, T_pred, num_modes * 6]
    # Reshape to [batch, T_pred, num_modes, 6]
    raw_output = raw_output.view(raw_output.size(0), raw_output.size(1), num_modes, 6)
    # Split into GMM parameters
    mu_x = raw_output[..., 0]
    mu_y = raw_output[..., 1]
    sigma_x = torch.exp(raw_output[..., 2])  # positive
    sigma_y = torch.exp(raw_output[..., 3])  # positive
    rho = torch.tanh(raw_output[..., 4])     # between -1 and 1
    log_pi = raw_output[..., 5]              # mixture weights logits
    pi = F.softmax(log_pi, dim=-1)           # normalize to probabilities
    # Return result
    return mu_x, mu_y, sigma_x, sigma_y, rho, pi

def load_model(MODEL_CLASS, model_path, device, prediction_length, multimodal):
    if multimodal=="unimodal":
        model = MODEL_CLASS(prediction_length=prediction_length)
    elif multimodal=="multimodal_gmm":
        model = MODEL_CLASS(prediction_length=prediction_length, gmm=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_model_scratch(MODEL_CLASS, device, prediction_length, multimodal):
    if multimodal=="unimodal":  
        model = MODEL_CLASS(prediction_length=prediction_length)
    elif multimodal=="multimodal_gmm":
        model = MODEL_CLASS(prediction_length=prediction_length, gmm=True)
    model.to(device)
    return model
