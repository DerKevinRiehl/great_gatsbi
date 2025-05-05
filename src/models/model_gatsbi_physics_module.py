"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of GATsBi's physics model, that leverages
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
    
class GATsBi_Physics_Module(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, gmm=False, num_modes=5):
        super(GATsBi_Physics_Module, self).__init__()
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
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # fusion
        self.fusion_decoder = nn.LSTM(hidden_dim*4, hidden_dim, batch_first=True)
            # final output layer
        if not gmm:
            self.output = output_decoding_layer_unimodal(hidden_dim, 1, output_dim)
        else:
            self.output = output_decoding_layer_multimodal_gmm(hidden_dim, num_modes)

    def encode_physics_predictions(self, t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk):      
        # Encode constant velocity predictions
        _, (h_pred_cv, _) = self.cv_encoder(t_pred_cv)
        h_pred_cv = h_pred_cv[-1]  # [batch, hidden_dim]
        # Encode constant acceleration predictions
        _, (h_pred_ca, _) = self.ca_encoder(t_pred_ca)
        h_pred_ca = h_pred_ca[-1]  # [batch, hidden_dim]
        # Encode constant bicycle kinematics predictions
        _, (h_pred_bk, _) = self.bk_encoder(t_pred_bk)
        h_pred_bk = h_pred_bk[-1]  # [batch, hidden_dim]
        # Encode constant xkalman filter predictions
        _, (h_pred_xk, _) = self.xk_encoder(t_pred_xk)
        h_pred_xk = h_pred_xk[-1]  # [batch, hidden_dim]
        return h_pred_cv, h_pred_ca, h_pred_bk, h_pred_xk
    
    def forward(self, t_ego_hist, t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk):
        """
        ego_hist: [batch, hist_len, 2] – past positions
        pred_cv:  [batch, pred_len, 2] – CV prediction for the future
        pred_ca:  [batch, pred_len, 2] – CA prediction for the future
        pred_bk:  [batch, pred_len, 2] – BK prediction for the future
        pred_xk:  [batch, pred_len, 2] – XK prediction for the future
        """
        # Encode physics predictions to H-Domain
        h_pred_cv, h_pred_ca, h_pred_bk, h_pred_xk = self.encode_physics_predictions(t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk)
        
        # Fusion
        h_context = torch.cat([h_pred_cv, h_pred_ca, h_pred_bk, h_pred_xk], dim=1)  # [batch, hidden*5]        
        h_context_repeated = h_context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]  
        h_context_fused, _ = self.fusion_decoder(h_context_repeated)
    
        # Output Layer
        if not self.gmm:
            return output_decode_unimodal(h_context_fused, self.output)
        else:
            return output_decode_multimodal_gmm(h_context_fused, self.num_modes, self.output)
