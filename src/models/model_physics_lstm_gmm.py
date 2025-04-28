"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of a physics-informed LSTM, that leverages
not only the ego vehicle's historical trajectory, but also the forecasts from
constant velocity, constant acceleration, bicycle-kinematic, and extended-Kalman-filtering prediction models.

This implementation is a multi-modal future forecasting using Gaussian Mixture Model (GMM).
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F




# """
# #############################################################################
# ### MODEL
class PhysicsLSTM_GMM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, num_modes=5):
        super(PhysicsLSTM_GMM, self).__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.num_modes = num_modes  # New: number of Gaussians

        # --- encoders (unchanged) ---
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # --- decoder (unchanged) ---
        self.decoder_lstm = nn.LSTM(hidden_dim * 5 + 6 + input_dim, hidden_dim, batch_first=True)

        # --- output layer (CHANGED) ---
        # Instead of 2 outputs (x, y), we predict for each mode:
        # (mu_x, mu_y, sigma_x, sigma_y, correlation rho) + mixture weight
        self.output_layer = nn.Linear(hidden_dim, num_modes * 6)  # 6 params per mode

    def forward(self, ego_hist, pred_cv, pred_ca, pred_bk, pred_xk):
        """
        ego_hist: [batch, hist_len, 2]
        pred_cv, pred_ca, pred_bk, pred_xk: [batch, pred_len, 2]
        """
        # --- encoders (unchanged) ---
        _, (h_hist, _) = self.hist_encoder(ego_hist)
        h_hist = h_hist[-1]

        _, (h_cv, _) = self.cv_encoder(pred_cv)
        h_cv = h_cv[-1]

        _, (h_ca, _) = self.ca_encoder(pred_ca)
        h_ca = h_ca[-1]

        _, (h_bk, _) = self.bk_encoder(pred_bk)
        h_bk = h_bk[-1]

        _, (h_xk, _) = self.xk_encoder(pred_xk)
        h_xk = h_xk[-1]

        # --- decoder input ---
        context = torch.cat([h_hist, h_cv, h_ca, h_bk, h_xk], dim=1)
        context_repeated = context.unsqueeze(1).repeat(1, self.prediction_length, 1)

        decoder_input = torch.cat([context_repeated, pred_cv, pred_ca, pred_bk, pred_xk], dim=2)

        # --- decode ---
        decoder_output, _ = self.decoder_lstm(decoder_input)

        # --- output head (CHANGED) ---
        raw_output = self.output_layer(decoder_output)  # [batch, T_pred, num_modes * 6]

        # Reshape to [batch, T_pred, num_modes, 6]
        raw_output = raw_output.view(raw_output.size(0), raw_output.size(1), self.num_modes, 6)

        # Split into parameters
        mu_x = raw_output[..., 0]
        mu_y = raw_output[..., 1]
        sigma_x = torch.exp(raw_output[..., 2])  # positive
        sigma_y = torch.exp(raw_output[..., 3])  # positive
        rho = torch.tanh(raw_output[..., 4])     # between -1 and 1
        log_pi = raw_output[..., 5]              # mixture weights logits

        pi = F.softmax(log_pi, dim=-1)           # normalize to probabilities

        return mu_x, mu_y, sigma_x, sigma_y, rho, pi
# """

def load_physics_lstm_gmm_model(model_path, device, prediction_length):
    model = PhysicsLSTM_GMM(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
