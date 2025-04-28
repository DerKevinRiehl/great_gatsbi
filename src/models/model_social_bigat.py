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

"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn




# #############################################################################
# ### MODEL

class SocialBiGAT(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, gat_heads=4):
        super(SocialBiGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.gat_heads = gat_heads

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

        # Output MLP: refines decoder outputs
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

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

        # Apply output MLP
        out = self.output_mlp(h_dec)  # [batch, pred_len, 2]

        return out

def load_social_bigat_model(model_path, device, prediction_length):
    model = SocialBiGAT(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
