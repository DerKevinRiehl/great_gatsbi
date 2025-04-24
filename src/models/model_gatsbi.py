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
        # Params

        # Network Structure
            # linear input layers
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_features, bias=False)
        self.a = nn.Linear(3 * out_features, 1, bias=False)  # includes edge feature
            # leaky relu
        self.leakyrelu = nn.LeakyReLU(alpha)
            # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_attr):
        """
        h: [B, N, F]                 - node features
        edge_attr: [B, N, N, E]      - edge features
        """
        B, N, _ = h.size()
        Wh = self.W(h)  # [B, N, F_out]

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)   # [B, N, N, F_out]
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)   # [B, N, N, F_out]
        We = self.edge_proj(edge_attr)             # [B, N, N, F_out]

        a_input = torch.cat([Wh_i, Wh_j, We], dim=-1)  # [B, N, N, 3 * F_out]
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [B, N, N]

        # Optional masking based on edge existence
        mask = (edge_attr.abs().sum(dim=-1) == 0)  # [B, N, N]
        e = e.masked_fill(mask, float('-inf'))

        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.bmm(attention, Wh)  # [B, N, F_out]
        return h_prime, attention

class GATSBI(nn.Module):  # V2
    def __init__(self, input_dim=2, hidden_dim=64, gat_out_dim=64, prediction_length=25):
        super(GATSBI, self).__init__()
        # Params
        self.pred_len = prediction_length
        # Network Structure
            # encoder LSTM for each agent's history
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # gat for extracting latent social forces
        self.gat = GATLayerWithEdgeFeatures(hidden_dim, gat_out_dim, edge_dim=4)
            # decoder LSTM for predicting future positions
        self.decoder = nn.LSTM(gat_out_dim * 2 + hidden_dim, hidden_dim, batch_first=True)
            # final output layer (e.g., predicting delta x, y)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        # self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, ego_hist, neighbor_histories, adj, dist):
        """
        ego_hist: [B, T_hist, 2]              - history of ego agent
        neighbor_histories: [B, N, T_hist, 2] - history of neighbors
        adj: [B, N+1, N+1, E]                 - adjacency matrix for all agents (neighbors + ego)
        dist: [B, 1, 1]                       - distance from road boder for ego agent
        """
        B, N, T, _ = neighbor_histories.shape
    
        # 1. Encode ego history
        _, (h_ego, _) = self.encoder(ego_hist)  # h_ego: [1, B, hidden]
        h_ego = h_ego.squeeze(0)                # [B, hidden]
    
        # 2. Encode each neighbor history
        encoded_neighbors = []
        for i in range(N):
            _, (h, _) = self.encoder(neighbor_histories[:, i])  # [1, B, hidden]
            encoded_neighbors.append(h.squeeze(0))              # [B, hidden]
        encoded_neighbors = torch.stack(encoded_neighbors, dim=1)  # [B, N, hidden]
    
        # 3. Stack neighbors and ego (ego LAST)
        all_agents = torch.cat([encoded_neighbors, h_ego.unsqueeze(1)], dim=1)  # [B, N+1, hidden]
        #        neighbors (0..N-1), ego (N)
    
        # 4. GAT over all agents (ego + neighbors)
        h_gat, attn = self.gat(all_agents, adj)  # [B, N+1, gat_out_dim], attn: [B, N+1, N+1]
    
        """
        # 5. Pool neighbors (excluding ego) for context
        # Ego is at index N, neighbors are 0..N-1
        pooled_neighbors = torch.sum(
            attn[:, N, :N].unsqueeze(-1) * h_gat[:, :N], dim=1
        )  # [B, gat_out_dim]
        """
        
        # """
        # 5. Pool all agents (including ego) for context
        # Ego is at index N, neighbors are 0..N-1
        # Use ego's attention over all agents (including itself)
        pooled_agents = torch.sum(
            attn[:, N, :, None] * h_gat, dim=1
        )  # [B, gat_out_dim]
        # """
        
        # 6. Combine ego GAT, pooled context, and ego encoding for decoder input
        # decoder_input = torch.cat([h_gat[:, N], pooled_neighbors, h_ego], dim=1)  # [B, gat_out_dim*2 + hidden_dim]
        decoder_input = torch.cat([h_gat[:, N], pooled_agents, h_ego], dim=1)  # [B, gat_out_dim*2 + hidden_dim]
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, T_pred, ...]
        # 7. Decode future trajectory
        h_dec, _ = self.decoder(decoder_input)
        out = self.output(h_dec)  # [B, T_pred, 2]
        
        # return
        return out, attn  # attn: [B, N+1, N+1]
    
def load_gatsbi_model(model_path, device, prediction_length):
    model = GATSBI(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
