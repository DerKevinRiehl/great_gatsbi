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
from models.model_utils import output_decoding_layer_unimodal, output_decoding_layer_multimodal_gmm
from models.model_utils import output_decode_unimodal, output_decode_multimodal_gmm




# #############################################################################
# ### MODEL

class SocialBiGAT(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, gat_heads=4, gmm=False, num_modes=5):
        super(SocialBiGAT, self).__init__()
        # ### PARAMS
            # general
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gmm = gmm
        self.num_modes = num_modes
            # model specific
        self.gat_heads = gat_heads
        # ### NETWORK STRUCTURE
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
            # fusion
        self.fusion_decoder = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True)
            # final output layer
        if not gmm:
            self.output = output_decoding_layer_unimodal(hidden_dim, 5, output_dim)
        else:
            self.output = output_decoding_layer_multimodal_gmm(hidden_dim, num_modes)

    def forward(self, t_ego_hist, t_neighbor_hist):
        """
        ego_hist: [batch, history_length, 2]
        neighbor_hists: [batch, num_neighbors, history_length, 2]
        """
        batch_size, num_neighbors, history_length, _ = t_neighbor_hist.shape

        # Encode ego history
        _, (h_ego, _) = self.encoder(t_ego_hist)  # [1, batch, hidden_dim]
        h_ego = h_ego.squeeze(0)  # [batch, hidden_dim]

        # Encode each neighbor
        h_neighbors = []
        for i in range(num_neighbors):
            hist = t_neighbor_hist[:, i]  # [batch, history_length, 2]
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

        # Fusion
        h_context = torch.cat([h_ego, attended], dim=1)  # [batch, hidden*5]        
        h_context_repeated = h_context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]  
        h_context_fused, _ = self.fusion_decoder(h_context_repeated)
    
        # Output Layer
        if not self.gmm:
            return output_decode_unimodal(h_context_fused, self.output)
        else:
            return output_decode_multimodal_gmm(h_context_fused, self.num_modes, self.output)
