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
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
from models.model_utils import output_decoding_layer_unimodal, output_decoding_layer_multimodal_gmm
from models.model_utils import output_decode_unimodal, output_decode_multimodal_gmm




# #############################################################################
# ### MODEL
    
class EgoLSTM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, gmm=False, num_modes=5):
        super(EgoLSTM, self).__init__()
        # ### PARAMS
            # general
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gmm = gmm
        self.num_modes = num_modes
            # model specific
        # ### NETWORK STRUCTURE
            # Separate encoders for trajectory history and physics prediction
        self.ego_hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # fusion
        self.fusion_decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            # final output layer
        if not gmm:
            self.output = output_decoding_layer_unimodal(hidden_dim, 1, output_dim)
        else:
            self.output = output_decoding_layer_multimodal_gmm(hidden_dim, num_modes)

    def encode_ego_history(self, t_ego_hist):
        _, (h_ego_hist, _) = self.ego_hist_encoder(t_ego_hist)
        h_ego_hist = h_ego_hist[-1]
        return h_ego_hist # [batch, hidden_dim]
    
    def forward(self, t_ego_hist):
        """
        ego_hist: [batch, hist_len, 2] – past positions
        """
        # Encode ego history to H-Domain
        h_ego_hist = self.encode_ego_history(t_ego_hist)

        # Fusion
        h_context = torch.cat([h_ego_hist], dim=1)  # [batch, hidden*5]        
        h_context_repeated = h_context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]  
        h_context_fused, _ = self.fusion_decoder(h_context_repeated)
        
        # Output Layer
        if not self.gmm:
            return output_decode_unimodal(h_context_fused, self.output)
        else:
            return output_decode_multimodal_gmm(h_context_fused, self.num_modes, self.output)
