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
import numpy

# from models.model_classic import ModelClassic, constant_velocity_predictor

# #############################################################################
# ### MODEL

# GAT Layer with Edge Features and LayerNorm
class GATLayerWithEdgeFeatures(nn.Module):
    def __init__(self, in_features, out_features, edge_dim=4, dropout=0.1, alpha=0.2):
        super(GATLayerWithEdgeFeatures, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_features, bias=False)
        self.a = nn.Linear(3 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_features)  # LayerNorm after GAT output

    def forward(self, node_features, edge_features):
        B, N, _ = node_features.size()
        Wh = self.W(node_features)  # [B, N, F_out]
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F_out]
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, F_out]
        We = self.edge_proj(edge_features)  # [B, N, N, F_out]

        a_input = torch.cat([Wh_i, Wh_j, We], dim=-1)  # [B, N, N, 3 * F_out]
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [B, N, N]

        # Mask non-existing edges
        mask = (edge_features.abs().sum(dim=-1) == 0)
        e = e.masked_fill(mask, float('-inf'))

        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # [B, N, F_out]
        h_prime = self.layernorm(h_prime)  # Apply LayerNorm
        return h_prime, attention

# Dynamic Decoder with LayerNorm and Dropout
class DynamicDecoderWithLayerNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(DynamicDecoderWithLayerNorm, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, prev_output=None):
        if prev_output is not None:
            context = torch.cat([context, prev_output], dim=-1)  # Concatenate previous output

        lstm_out, _ = self.lstm(context)  # [B, T_pred, hidden_dim]
        lstm_out = self.layernorm(lstm_out)  # Apply LayerNorm
        lstm_out = self.dropout(lstm_out)    # Apply Dropout
        output = self.output_layer(lstm_out)  # Final output [B, T_pred, 2]

        return output

def constant_velocity_predictor(hist, history_dt=0.04, prediction_length=50):
    """
    Predicts future x, y positions assuming constant velocity.
    
    Parameters:
        hist [32,100,2]
        history_dt (float): Time step between observations in seconds (default: 0.025)
        prediction_length (int): Number of future time steps to predict
        
    Returns:
        pred [32,100,2]
    """
    B, _, _ = hist.shape
    
    # estimate velocity from last two points (or use filtered velocity if available)
    n = 1
    vx = (hist[:,-1,0] - hist[:,-1-n,0])/ (n * history_dt)
    vy = (hist[:,-1,1] - hist[:,-1-n,1])/ (n * history_dt)
    vx = vx.unsqueeze(-1).repeat(1, prediction_length)
    vy = vy.unsqueeze(-1).repeat(1, prediction_length)
    
    # predict future positions
    future_times = torch.arange(1, prediction_length + 1) * history_dt
    future_times = future_times.repeat(B, 1)
   
    x_pred = hist[:,-1,0].unsqueeze(-1).repeat(1, prediction_length) + vx * future_times
    y_pred = hist[:,-1,1].unsqueeze(-1).repeat(1, prediction_length) + vy * future_times
    pred = torch.cat((x_pred, y_pred), dim=-1)

    return pred

# GATSBI Model with Two GAT Layers and Dynamic Decoder
class GATSBIv3(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, gat_out_dim=64, prediction_length=25):
        super(GATSBIv3, self).__init__()
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.pred_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.dist_encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.agent_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # GAT Layers
        self.gat1 = GATLayerWithEdgeFeatures(hidden_dim, gat_out_dim, edge_dim=4)
        self.gat2 = GATLayerWithEdgeFeatures(gat_out_dim, gat_out_dim, edge_dim=4)
        
        # Dynamic Decoder
        decoder_input_dim = gat_out_dim 
        self.decoder = DynamicDecoderWithLayerNorm(decoder_input_dim, hidden_dim)

    def encode_agent_futures(self, ego_pred, neighbor_hists):
        """Encode ego and neighbor future predictions separately."""
        _, (h_ego, _) = self.pred_encoder(ego_pred)
        h_ego = h_ego.squeeze(0)  # [B, hidden_dim]

        B, N, T, _ = neighbor_hists.shape
        
        neighbor_encodings = []
        for i in range(N):
            neighbor_preds = constant_velocity_predictor(neighbor_hists[:,i,:,:], prediction_length=self.prediction_length)
            # neighbor_preds = ModelClassic(model_func=constant_velocity_predictor, prediction_length=self.prediction_length)(neighbor_hists)
            _, (h_neigh, _) = self.pred_encoder(neighbor_preds[:, i])  # [1, B, hidden_dim]
            neighbor_encodings.append(h_neigh.squeeze(0))
        neighbor_encodings = torch.stack(neighbor_encodings, dim=1)  # [B, N, hidden_dim]

        return h_ego, neighbor_encodings

    def encode_agent_histories(self, ego_hist, neighbor_hists):
        """Encode ego and neighbor histories separately."""
        B, N, T, _ = neighbor_hists.shape
        _, (h_ego, _) = self.agent_encoder(ego_hist)
        h_ego = h_ego.squeeze(0)  # [B, hidden_dim]

        neighbor_encodings = []
        for i in range(N):
            _, (h_neigh, _) = self.agent_encoder(neighbor_hists[:, i])  # [1, B, hidden_dim]
            neighbor_encodings.append(h_neigh.squeeze(0))
        neighbor_encodings = torch.stack(neighbor_encodings, dim=1)  # [B, N, hidden_dim]

        return h_ego, neighbor_encodings

    def encode_road_features(self, dist):
        """Encode road feature sequence."""
        dist = dist.unsqueeze(-1)  # [B, SeqLen, 1]
        _, (h_dist, _) = self.dist_encoder(dist)
        h_dist = h_dist.squeeze(0)  # [B, hidden_dim]
        return h_dist
    
    def forward(self, ego_hist, neighbor_hists, adj, pred_cv, dist):
        """
        Forward pass of the GATSBI model.
            ego_hist       - [32, 100, 2]
            neighbor_hists - [32, 5, 100, 2]
            adj            - [32, 6, 6, 4]
            pred_cv        - [32, 25, 2]
            dist           - [32, 100]
        """
        # History Encoding for Ego and Neighbors
        ego_hist_encodings, neighbor_hist_encodings = self.encode_agent_histories(ego_hist, neighbor_hists)
        all_agents_hist = torch.cat([neighbor_hist_encodings, ego_hist_encodings.unsqueeze(1)], dim=1)  # [B, N+1, hidden_dim]

        # Future Encoding for Ego and Neighbors
        ego_fut_encodings, neighbor_fut_encodings = self.encode_agent_futures(pred_cv, neighbor_hists)
        all_agents_fut = torch.cat([neighbor_fut_encodings, ego_fut_encodings.unsqueeze(1)], dim=1)  # [B, N+1, hidden_dim]

        # Road Encoding
        context_road = self.encode_road_features(dist)
        context_repeated_road = context_road.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [B, T_pred, hidden_dim]

        # Social Encoding
        node_features = torch.cat([all_agents_hist, all_agents_fut, context_repeated_road], dim=1) # [B, 2*(N+1)+T_pred, hidden_dim]
        edge_features = adj

        h_gat1, attn1 = self.gat1(node_features, edge_features)  # First GAT layer
        h_gat2, attn2 = self.gat2(h_gat1, edge_features)  # Second GAT layer
        
        ego_attention = attn2[:, -1, :]  # Attention from second GAT layer
        context_social = torch.sum(ego_attention.unsqueeze(-1) * h_gat2, dim=1)  # [B, gat_out_dim]
        context_repeated_social = context_social.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [B, T_pred, gat_out_dim]

        # Decode
        decoder_output = self.decoder(context_repeated_social)  # [B, T_pred, 2]

        return decoder_output, attn2

    
def load_gatsbi_modelv3(model_path, device, prediction_length):
    model = GATSBIv3(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
