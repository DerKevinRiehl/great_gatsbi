"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of Social-LSTM following Alahi et al. 2016 (developed for pedestrians):
    @inproceedings{alahi2016social,
        title={Social lstm: Human trajectory prediction in crowded spaces},
        author={Alahi, Alexandre and Goel, Kratarth and Ramanathan, Vignesh and Robicquet, Alexandre and Fei-Fei, Li and Savarese, Silvio},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={961--971},
        year={2016}
    }
    
    This implementation is a multi-modal future forecasting using Gaussian Mixture Model (GMM).
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F




# #############################################################################
# ### MODEL
class SocialLSTM_GMM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, grid_size=(10, 10), pooling_radius=20.0, num_modes=5):
        super(SocialLSTM_GMM, self).__init__()
        # Params
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.pooling_radius = pooling_radius
        self.prediction_length = prediction_length
        self.num_modes = num_modes  # New: number of Gaussians
        # Network Structure
            # encoder LSTM for each agent's history
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # decoder LSTM for predicting future positions
        self.decoder = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, batch_first=True)  # with pooled hidden states
        # --- output layer (CHANGED) ---
        # Instead of 2 outputs (x, y), we predict for each mode:
        # (mu_x, mu_y, sigma_x, sigma_y, correlation rho) + mixture weight
        self.output_layer = nn.Linear(hidden_dim, num_modes * 6)  # 6 params per mode

    def social_pooling(self, ego_pos, all_hidden_states, all_positions):
        """
        Aggregate hidden states of nearby agents within pooling_radius
        ego_pos: [batch_size, 2]
        all_hidden_states: [batch_size, num_agents, hidden_dim]
        all_positions: [batch_size, num_agents, 2]
        """
        batch_size, num_agents, _ = all_positions.shape
        pooled = torch.zeros((batch_size, self.hidden_dim), device=ego_pos.device)
        for i in range(num_agents):
            dist = torch.norm(all_positions[:, i] - ego_pos, dim=1)
            mask = dist < self.pooling_radius
            pooled += mask[:, None] * all_hidden_states[:, i]  # broadcast over hidden dim
        return pooled

    def forward(self, ego_hist, neighbor_hists):
        """
        ego_hist: [batch, history_length, 2] – history of ego
        ego_pos: [batch, 2] – current position of ego
        neighbor_hists: [batch, num_neighbors, history_length, 2]
        neighbor_pos: [batch, num_neighbors, 2]
        """
        batch_size, num_neighbors, history_length, _ = neighbor_hists.shape
        # Encode ego history
        _, (h_ego, _) = self.encoder(ego_hist)  # [1, batch, hidden]
        h_ego = h_ego.squeeze(0)
        # Encode each neighbor
        h_neighbors = []
        for i in range(num_neighbors):
            hist = neighbor_hists[:, i]  # [batch, history_length, 2]
            _, (h, _) = self.encoder(hist)
            h_neighbors.append(h.squeeze(0))
        h_neighbors = torch.stack(h_neighbors, dim=1)  # [batch, num_neighbors, hidden]
        # Social pooling
        ego_pos = ego_hist[:,-1,:]
        neighbor_pos = neighbor_hists[:,:,-1,:]
        pooled_social = self.social_pooling(ego_pos, h_neighbors, neighbor_pos)  # [batch, hidden]
        # Decode with pooled context
        h_dec_in = torch.cat([h_ego, pooled_social], dim=1).unsqueeze(1).repeat(1, self.prediction_length, 1)
        h_dec, _ = self.decoder(h_dec_in)
        
        # --- output head (CHANGED) ---
        raw_output = self.output_layer(h_dec)  # [batch, T_pred, num_modes * 6]

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

def load_social_lstm_gmm_model(model_path, device, prediction_length):
    model = SocialLSTM_GMM(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
