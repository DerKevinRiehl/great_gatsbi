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




# #############################################################################
# ### MODEL
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
    def __init__(self, input_dim=2, hidden_dim=64, gat_out_dim=64, prediction_length=25):
        super(GATSBIv1, self).__init__()
        # Params
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim
        # Network Structure
            # Encoders
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dist_encoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.dist_proj = nn.Linear(hidden_dim, hidden_dim * 5)  # Project road feature to match physics feature
        self.agent_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # GAT
        self.gat = GATLayerWithEdgeFeatures(hidden_dim, gat_out_dim, edge_dim=4)
        # Decoder
        decoder_input_dim = gat_out_dim + hidden_dim * 5 + hidden_dim * 5
        self.decoder = nn.LSTM(decoder_input_dim, hidden_dim, batch_first=True)
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def encode_agent_histories(self, ego_hist, neighbor_hists):
        """Encode ego and neighbor histories separately."""
        B, N, T, _ = neighbor_hists.shape
        
        # Encode ego
        _, (h_ego, _) = self.agent_encoder(ego_hist)  # [1, B, hidden_dim]
        h_ego = h_ego.squeeze(0)                      # [B, hidden_dim]

        # Encode neighbors
        neighbor_encodings = []
        for i in range(N):
            _, (h_neigh, _) = self.agent_encoder(neighbor_hists[:, i])  # [1, B, hidden_dim]
            neighbor_encodings.append(h_neigh.squeeze(0))
        neighbor_encodings = torch.stack(neighbor_encodings, dim=1)     # [B, N, hidden_dim]

        return h_ego, neighbor_encodings

    def encode_physics_predictions(self, pred_cv, pred_ca, pred_bk, pred_xk, ego_hist):
        """Encode different physics-based future predictions."""
        _, (h_hist, _) = self.hist_encoder(ego_hist)
        _, (h_cv, _) = self.cv_encoder(pred_cv)
        _, (h_ca, _) = self.ca_encoder(pred_ca)
        _, (h_bk, _) = self.bk_encoder(pred_bk)
        _, (h_xk, _) = self.xk_encoder(pred_xk)

        # Collect last hidden states
        physics_context = torch.cat([
            h_hist[-1], h_cv[-1], h_ca[-1], h_bk[-1], h_xk[-1]
        ], dim=-1)  # [B, hidden_dim * 5]

        return physics_context

    def encode_road_features(self, dist):
        """Encode road feature sequence."""
        dist = dist.unsqueeze(-1)  # [B, SeqLen, 1]
        _, (h_dist, _) = self.dist_encoder(dist)
        h_dist = h_dist.squeeze(0)  # [B, hidden_dim]
        h_dist = self.dist_proj(h_dist)  # [B, hidden_dim*5] <--- PROJECT IT
        return h_dist

    def forward(self, ego_hist, neighbor_hists, adj, pred_cv, pred_ca, pred_bk, pred_xk, dist):
        """
        Forward pass of the GATSBI model.
            ego_hist       - [32, 100, 2]
            neighbor_hists - [32, 5, 100, 2]
            adj            - [32, 6, 6, 4]
            pred_cv        - [32, 25, 2]
            pred_ca        - [32, 25, 2]
            pred_bk        - [32, 25, 2]
            pred_xk        - [32, 25, 2]
            hist           - [32, 100]
        """
        # Social Encoding
        h_ego, neighbor_encodings = self.encode_agent_histories(ego_hist, neighbor_hists)
        all_agents = torch.cat([neighbor_encodings, h_ego.unsqueeze(1)], dim=1)  # [B, N+1, hidden_dim]
        node_features = all_agents
        edge_features = adj

        h_gat, attn = self.gat(node_features, edge_features)  # [B, N+1, gat_out_dim]
        
        ego_attention = attn[:, -1, :]  # [B, N+1]
        context_social = torch.sum(ego_attention.unsqueeze(-1) * h_gat, dim=1)  # [B, gat_out_dim]
        context_repeated_social = context_social.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [B, T_pred, gat_out_dim]

        # Physics Encoding
        context_physics = self.encode_physics_predictions(pred_cv, pred_ca, pred_bk, pred_xk, ego_hist)
        context_repeated_physics = context_physics.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [B, T_pred, hidden_dim*5]

        # Road Encoding
        context_road = self.encode_road_features(dist)
        context_repeated_road = context_road.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [B, T_pred, hidden_dim*5]

        # Concatenate all contexts
        decoder_input = torch.cat([
            context_repeated_social,  # [B, T_pred, gat_out_dim]
            context_repeated_physics, # [B, T_pred, hidden_dim*5]
            context_repeated_road     # [B, T_pred, hidden_dim*5]
        ], dim=-1)  # [B, T_pred, combined_dim]

        # Decode
        decoder_output, _ = self.decoder(decoder_input)  # [B, T_pred, hidden_dim]
        output = self.output_layer(decoder_output)       # [B, T_pred, 2]

        return output, attn
    
def load_gatsbi_modelv1(model_path, device, prediction_length):
    model = GATSBIv1(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
