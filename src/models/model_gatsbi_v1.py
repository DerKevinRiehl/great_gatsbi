"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of GATsBI model as part of this project.
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import output_decoding_layer_unimodal, output_decoding_layer_multimodal_gmm
from models.model_utils import output_decode_unimodal, output_decode_multimodal_gmm




# #############################################################################
# ### MODEL

def constant_velocity_predictor(hist, history_dt=0.04, prediction_length=50):
    """
    Predicts future x, y positions assuming constant velocity.
    
    Parameters:
        hist [B, N, T_hist, 2]
        history_dt (float): Time step between observations in seconds (default: 0.025)
        prediction_length (int): Number of future time steps to predict
        
    Returns:
        pred [B, N, T_pred, 2]
    """
    B, N, _, _ = hist.shape
    device = hist.get_device()
    device = torch.device("cpu" if device == -1 else f"cuda:{device}")
    
    # estimate velocity from last two points (or use filtered velocity if available)
    n = 1
    vx = (hist[:,:,-1,0] - hist[:,:,-1-n,0])/ (n * history_dt)
    vy = (hist[:,:,-1,1] - hist[:,:,-1-n,1])/ (n * history_dt)
    vx = vx.unsqueeze(-1).repeat(1, 1, prediction_length)
    vy = vy.unsqueeze(-1).repeat(1, 1, prediction_length)
    
    # predict future positions
    future_times = torch.arange(1, prediction_length + 1) * history_dt
    future_times = future_times.repeat(B, N, 1).to(device=device)
   
    x_pred = hist[:,:,-1,0].unsqueeze(-1).repeat(1, 1, prediction_length) + vx * future_times
    y_pred = hist[:,:,-1,1].unsqueeze(-1).repeat(1, 1, prediction_length) + vy * future_times
    pred = torch.cat((x_pred.unsqueeze(-1), y_pred.unsqueeze(-1)), dim=-1)
    return pred

class GATLayerWithEdgeFeatures(nn.Module):
    def __init__(self, in_features, out_features, edge_dim=4, dropout=0.1, alpha=0.2):
        super(GATLayerWithEdgeFeatures, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_features, bias=False)
        self.a = nn.Linear(3 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, edge_features):
        """
        node_features:    [B, N, F] - node features
        edge_features: [B, N, N, E] - edge features
        """
        B, N, _ = node_features.size()
        Wh = self.W(node_features)  # [B, N, F_out]

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)   # [B, N, N, F_out]
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)   # [B, N, N, F_out]
        We = self.edge_proj(edge_features)              # [B, N, N, F_out]

        a_input = torch.cat([Wh_i, Wh_j, We], dim=-1)  # [B, N, N, 3 * F_out]
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [B, N, N]

        # Mask non-existing edges (if edge_attr == 0)
        mask = (edge_features.abs().sum(dim=-1) == 0)
        e = e.masked_fill(mask, float('-inf'))

        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # [B, N, F_out]
        return h_prime, attention

class GATSBIv1(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, gat_out_dim=64, output_dim=2, prediction_length=25, gmm=False, num_modes=5):
        super(GATSBIv1, self).__init__()
        # ### PARAMS
            # general
        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gmm = gmm
        self.num_modes = num_modes
            # model specific
        self.gat_out_dim = gat_out_dim        
        # ### NETWORK STRUCTURE
            # Encoders
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.agent_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.pred_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # Input MLP: refines encoded LSTM features before GAT
        self.lstm_mlp_refinement_physics = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lstm_mlp_refinement_social = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
            # GAT
        self.gat = GATLayerWithEdgeFeatures(hidden_dim*2, gat_out_dim, edge_dim=4)
            # fusion
        self.fusion_decoder = nn.LSTM(gat_out_dim + hidden_dim*1, hidden_dim, batch_first=True)
            # final output layer
        if not gmm:
            self.output = output_decoding_layer_unimodal(hidden_dim, 5, output_dim)
        else:
            self.output = output_decoding_layer_multimodal_gmm(hidden_dim, num_modes)

    def encode_agent_features(self, t_ego_hist, t_neighbor_hist):
        """Encode ego and neighbor histories separately."""
        B, N, T, _ = t_neighbor_hist.shape
        
        # Encode ego
        _, (h_ego_hist, _) = self.agent_encoder(t_ego_hist)  # [1, B, hidden_dim]
        h_ego_hist = h_ego_hist.squeeze(0)                      # [B, hidden_dim]
        # h_ego_hist = self.lstm_mlp_refinement_social(h_ego_hist)

        # Encode neighbors
        h_neighbor_hist = []
        for i in range(N):
            _, (h_neigh, _) = self.agent_encoder(t_neighbor_hist[:, i])  # [1, B, hidden_dim]
            h_neighbor_hist.append(h_neigh.squeeze(0))
        h_neighbor_hist = torch.stack(h_neighbor_hist, dim=1)     # [B, N, hidden_dim]
        # h_neighbor_hist = self.lstm_mlp_refinement_social(h_neighbor_hist)

        # Encode egos' anticipated trajectory
        t_ego_pred = constant_velocity_predictor(t_ego_hist.unsqueeze(1), prediction_length=self.prediction_length)
        t_ego_pred = t_ego_pred.squeeze(1)
        _, (h_ego_pred, _) = self.pred_encoder(t_ego_pred)  # [1, B, hidden_dim]
        # h_ego_pred = self.lstm_mlp_refinement_social(h_ego_pred)
        h_ego_pred = h_ego_pred.permute(1, 0, 2)  # shape becomes [32, 1, 64]

        # Encode neighbors' anticipated trajectory
        t_neighbor_preds = constant_velocity_predictor(t_neighbor_hist, prediction_length=self.prediction_length)
        h_neighbor_preds = []
        for i in range(N):
            _, (h_neigh, _) = self.pred_encoder(t_neighbor_preds[:, i])  # [1, B, hidden_dim]
            h_neighbor_preds.append(h_neigh.squeeze(0))
        h_neighbor_preds = torch.stack(h_neighbor_preds, dim=1)  # [B, N, hidden_dim]
        # h_neighbor_preds = self.lstm_mlp_refinement_social(h_neighbor_preds)

        return torch.cat([h_neighbor_hist, h_ego_hist.unsqueeze(1)], dim=1), torch.cat([h_neighbor_preds, h_ego_pred], dim=1), h_ego_hist

    def encode_physics_predictions(self, t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk):      
        # LSTM ENCODING
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
        
        # LSTM ENCODING REFINEMENT WITH MLP
        h_pred_cv = self.lstm_mlp_refinement_physics(h_pred_cv)
        h_pred_ca = self.lstm_mlp_refinement_physics(h_pred_ca)
        h_pred_bk = self.lstm_mlp_refinement_physics(h_pred_bk)
        h_pred_xk = self.lstm_mlp_refinement_physics(h_pred_xk)
        
        return torch.cat([h_pred_cv, h_pred_ca, h_pred_bk, h_pred_xk], dim=1)    

    def forward(self, t_ego_hist, t_neighbor_hist, s_adj, t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk):
        """
        Forward pass of the GATSBI model.
            t_ego_hist       - [32, 100, 2]
            t_neighbor_hist  - [32, 5, 100, 2]
            s_adj            - [32, 6, 6, 4]
            t_pred_cv        - [32, 25, 2]
            t_pred_ca        - [32, 25, 2]
            t_pred_bk        - [32, 25, 2]
            t_pred_xk        - [32, 25, 2]
        """
        
        # Physics Encoding
        # h_context_physics = self.encode_physics_predictions(t_pred_cv, t_pred_ca, t_pred_bk, t_pred_xk)
        
        # Social Encoding
        h_social_node_features_history, h_social_node_features_future, h_ego = self.encode_agent_features(t_ego_hist, t_neighbor_hist)
        h_social_node_features = torch.cat(
            [h_social_node_features_history, h_social_node_features_future], dim=-1
        )  # shape: [32, 6, 128]
        # print(">>")
        # print(h_social_node_features_history.shape)
        # print(h_social_node_features_future.shape)
        # print(h_social_node_features.shape)
        # print(h_ego.shape)
        # print(s_adj.shape)
        # print(">>")
        s_edge_features = s_adj
            # past
        h_gat, w_neighbor_attention = self.gat(h_social_node_features, s_edge_features)  # [B, N+1, gat_out_dim]        
        w_ego_attention = w_neighbor_attention[:, -1, :]  # [B, N+1]
        h_context_social = torch.sum(w_ego_attention.unsqueeze(-1) * h_gat, dim=1)  # [B, gat_out_dim]

        # Fusion
        h_context = torch.cat([h_ego, h_context_social], dim=1)#, h_context_physics], dim=1)  # [batch, hidden*5]        
        h_context_repeated = h_context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]  
        h_context_fused, _ = self.fusion_decoder(h_context_repeated)
    
        # Output Layer
        if not self.gmm:
            return output_decode_unimodal(h_context_fused, self.output), w_ego_attention
        else:
            return output_decode_multimodal_gmm(h_context_fused, self.num_modes, self.output), w_ego_attention
