"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of Social-BiGAT following Kosaraju et al. 2019 (developed for pedestrians):
    @article{kosaraju2019social,
      title={Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks},
      author={Kosaraju, Vineet and Sadeghian, Amir and Mart{\'\i}n-Mart{\'\i}n, Roberto and Reid, Ian and Rezatofighi, Hamid and Savarese, Silvio},
      journal={Advances in neural information processing systems},
      volume={32},
      year={2019}
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

class SocialBiGAT_GMM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, gat_heads=4, num_modes=5):
        super(SocialBiGAT_GMM, self).__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.gat_heads = gat_heads
        self.num_modes = num_modes  # New: number of Gaussians

        # Encode agent histories
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Input MLP: refines encoded LSTM features before GAT
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT layer: attention over neighbors
        self.gat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=gat_heads, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)

        # --- output layer (CHANGED) ---
        # Instead of 2 outputs (x, y), we predict for each mode:
        # (mu_x, mu_y, sigma_x, sigma_y, correlation rho) + mixture weight
        self.output_layer = nn.Linear(hidden_dim, num_modes * 6)  # 6 params per mode

    def forward(self, ego_hist, neighbor_hists):
        """
        ego_hist: [batch, history_length, 2]
        neighbor_hists: [batch, num_neighbors, history_length, 2]
        """
        batch_size, num_neighbors, history_length, _ = neighbor_hists.shape

        # Encode ego history
        _, (h_ego, _) = self.encoder(ego_hist)  # [1, batch, hidden_dim]
        h_ego = h_ego.squeeze(0)  # [batch, hidden_dim]

        # Encode each neighbor
        h_neighbors = []
        for i in range(num_neighbors):
            hist = neighbor_hists[:, i]  # [batch, history_length, 2]
            _, (h, _) = self.encoder(hist)
            h_neighbors.append(h.squeeze(0))  # [batch, hidden_dim]
        h_neighbors = torch.stack(h_neighbors, dim=1)  # [batch, num_neighbors, hidden_dim]

        # Apply input MLPs
        h_ego = self.input_mlp(h_ego)  # [batch, hidden_dim]
        h_neighbors = self.input_mlp(h_neighbors)  # [batch, num_neighbors, hidden_dim]

        # GAT: ego attends to neighbors
        query = h_ego.unsqueeze(1)  # [batch, 1, hidden_dim]
        key_value = h_neighbors  # [batch, num_neighbors, hidden_dim]

        attended, _ = self.gat(query, key_value, key_value)  # [batch, 1, hidden_dim]
        attended = attended.squeeze(1)  # [batch, hidden_dim]

        # Concatenate ego encoding + attended neighbor features
        h_dec_in = torch.cat([h_ego, attended], dim=1)  # [batch, hidden_dim * 2]
        h_dec_in = h_dec_in.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, pred_len, hidden_dim*2]

        # Decode future
        h_dec, _ = self.decoder(h_dec_in)  # [batch, pred_len, hidden_dim]

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

def load_social_bigat_gmm_model(model_path, device, prediction_length):
    model = SocialBiGAT_GMM(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
