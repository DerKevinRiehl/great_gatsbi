"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
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
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
from models.model_utils import output_decoding_layer_unimodal, output_decoding_layer_multimodal_gmm
from models.model_utils import output_decode_unimodal, output_decode_multimodal_gmm




# #############################################################################
# ### MODEL
class SocialLSTM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, grid_size=(10, 10), pooling_radius=20.0, gmm=False, num_modes=5):
        super(SocialLSTM, self).__init__()
        # ### PARAMS
            # general
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gmm = gmm
        self.num_modes = num_modes
            # model specific
        self.grid_size = grid_size
        self.pooling_radius = pooling_radius
        # ### NETWORK STRUCTURE
            # encoder LSTM for each agent's history
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # fusion
        self.fusion_decoder = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True)
            # final output layer
        if not gmm:
            self.output = output_decoding_layer_unimodal(hidden_dim, 5, output_dim)
        else:
            self.output = output_decoding_layer_multimodal_gmm(hidden_dim, num_modes)

    def social_pooling(self, t_ego_pos, h_neighbor_hist, t_neighbor_pos):
        """
        Aggregate hidden states of nearby agents within pooling_radius
        t_ego_pos: [batch_size, 2]
        h_neighbor_hist: [batch_size, num_agents, hidden_dim]
        t_neighbor_pos: [batch_size, num_agents, 2]
        """
        batch_size, num_agents, _ = t_neighbor_pos.shape
        h_pooled = torch.zeros((batch_size, self.hidden_dim), device=t_ego_pos.device)
        for i in range(num_agents):
            dist = torch.norm(t_neighbor_pos[:, i] - t_ego_pos, dim=1)
            mask = dist < self.pooling_radius
            h_pooled += mask[:, None] * h_neighbor_hist[:, i]  # broadcast over hidden dim
        return h_pooled

    def forward(self, t_ego_hist, t_neighbor_hist):
        """
        t_ego_hist: [batch, history_length, 2] – history of ego
        t_neighbor_hist: [batch, num_neighbors, history_length, 2]
        """
        batch_size, num_neighbors, history_length, _ = t_neighbor_hist.shape
        # Encode ego history
        _, (h_ego_hist, _) = self.encoder(t_ego_hist)  # [1, batch, hidden]
        h_ego_hist = h_ego_hist.squeeze(0)
        
        # Encode each neighbor
        h_neighbor_hist = []
        for i in range(num_neighbors):
            t_hist = t_neighbor_hist[:, i]  # [batch, history_length, 2]
            _, (h_hist, _) = self.encoder(t_hist)
            h_neighbor_hist.append(h_hist.squeeze(0))
        h_neighbor_hist = torch.stack(h_neighbor_hist, dim=1)  # [batch, num_neighbors, hidden]
        
        # Social pooling
        t_ego_pos = t_ego_hist[:,-1,:]
        t_neighbor_pos = t_neighbor_hist[:,:,-1,:]
        h_pooled_social = self.social_pooling(t_ego_pos, h_neighbor_hist, t_neighbor_pos)  # [batch, hidden]
    
        # Fusion
        h_context = torch.cat([h_ego_hist, h_pooled_social], dim=1)  # [batch, hidden*5]        
        h_context_repeated = h_context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]  
        h_context_fused, _ = self.fusion_decoder(h_context_repeated)
    
        # Output Layer
        if not self.gmm:
            return output_decode_unimodal(h_context_fused, self.output)
        else:
            return output_decode_multimodal_gmm(h_context_fused, self.num_modes, self.output)
