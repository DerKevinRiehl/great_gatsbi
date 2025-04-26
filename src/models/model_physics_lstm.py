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





"""
class PhysicsLSTM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2, dropout_prob=0.3):
        super(PhysicsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length

        # Apply Layer Normalization to inputs
        self.layer_norm_hist = nn.LayerNorm(input_dim)
        self.layer_norm_cv = nn.LayerNorm(input_dim)
        self.layer_norm_ca = nn.LayerNorm(input_dim)
        self.layer_norm_bk = nn.LayerNorm(input_dim)
        self.layer_norm_xk = nn.LayerNorm(input_dim)

        # Apply Dropout
        self.drop = nn.Dropout(dropout_prob)  # Apply dropout with a dropout probability

        # Separate encoders with Dropout
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)

        # Attention mechanism (optional but recommended)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 5, num_heads=4, batch_first=True)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_dim * 5 + 6 + input_dim, hidden_dim, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, ego_hist, pred_cv, pred_ca, pred_bk, pred_xk):
        # Normalize inputs
        ego_hist = self.layer_norm_hist(ego_hist)
        pred_cv = self.layer_norm_cv(pred_cv)
        pred_ca = self.layer_norm_ca(pred_ca)
        pred_bk = self.layer_norm_bk(pred_bk)
        pred_xk = self.layer_norm_xk(pred_xk)

        # Encode history
        _, (h_hist, _) = self.hist_encoder(ego_hist)
        h_hist = h_hist[-1]  # [batch, hidden_dim]

        # Encode physics-based predictions
        _, (h_cv, _) = self.cv_encoder(pred_cv)
        h_cv = h_cv[-1]  # [batch, hidden_dim]

        _, (h_ca, _) = self.ca_encoder(pred_ca)
        h_ca = h_ca[-1]  # [batch, hidden_dim]

        _, (h_bk, _) = self.bk_encoder(pred_bk)
        h_bk = h_bk[-1]  # [batch, hidden_dim]

        _, (h_xk, _) = self.xk_encoder(pred_xk)
        h_xk = h_xk[-1]  # [batch, hidden_dim]

        # Combine encodings and apply dropout
        context = torch.cat([h_hist, h_cv, h_ca, h_bk, h_xk], dim=1)  # [batch, hidden*5]
        context = self.drop(context)  # Apply dropout to the concatenated features

        # Apply attention (optional)
        context = context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*5]
        context, _ = self.attention(context, context, context)  # [batch, T_pred, hidden*5]

        # Concatenate physics predictions and context
        decoder_input = torch.cat([context, pred_cv, pred_ca, pred_bk, pred_xk], dim=2)

        # Decode
        decoder_output, _ = self.decoder_lstm(decoder_input)
        pred = self.output_layer(decoder_output)

        return pred
"""    

# """
# #############################################################################
# ### MODEL
class PhysicsLSTM(nn.Module):
    def __init__(self, prediction_length=25, input_dim=2, hidden_dim=64, output_dim=2):
        super(PhysicsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length

        # Separate encoders for trajectory history and physics prediction
        self.hist_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.cv_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ca_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.xk_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Decoder LSTM (conditioned on both encoded states + physics prediction at each timestep)
        self.decoder_lstm = nn.LSTM(hidden_dim * 5 +6+ input_dim, hidden_dim, batch_first=True)

        # Final output layer (e.g., delta x, y or absolute positions)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, ego_hist, pred_cv, pred_ca, pred_bk, pred_xk):
        """
        ego_hist: [batch, hist_len, 2] – past positions
        pred_cv:  [batch, pred_len, 2] – CV prediction for the future
        pred_ca:  [batch, pred_len, 2] – CA prediction for the future
        pred_bk:  [batch, pred_len, 2] – BK prediction for the future
        pred_xk:  [batch, pred_len, 2] – XK prediction for the future
        """
        # Encode history
        _, (h_hist, _) = self.hist_encoder(ego_hist)
        h_hist = h_hist[-1]  # [batch, hidden_dim]

        # Encode constant velocity predictions
        _, (h_cv, _) = self.cv_encoder(pred_cv)
        h_cv = h_cv[-1]  # [batch, hidden_dim]

        # Encode constant acceleration predictions
        _, (h_ca, _) = self.ca_encoder(pred_ca)
        h_ca = h_ca[-1]  # [batch, hidden_dim]

        # Encode constant bicycle kinematics predictions
        _, (h_bk, _) = self.bk_encoder(pred_bk)
        h_bk = h_bk[-1]  # [batch, hidden_dim]

        # Encode constant xkalman filter predictions
        _, (h_xk, _) = self.xk_encoder(pred_xk)
        h_xk = h_xk[-1]  # [batch, hidden_dim]
        
        # Prepare repeated context vector
        context = torch.cat([h_hist, h_cv, h_ca, h_bk, h_xk], dim=1)  # [batch, hidden*2]
        context_repeated = context.unsqueeze(1).repeat(1, self.prediction_length, 1)  # [batch, T_pred, hidden*2]

        # Concatenate CV predictions at each timestep
        decoder_input = torch.cat([context_repeated, pred_cv, pred_ca, pred_bk, pred_xk], dim=2)  # [batch, T_pred, hidden*2 + 2]

        # Decode
        decoder_output, _ = self.decoder_lstm(decoder_input)
        pred = self.output_layer(decoder_output)  # [batch, T_pred, 2]

        return pred
# """

def load_physics_lstm_model(model_path, device, prediction_length):
    model = PhysicsLSTM(prediction_length=prediction_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
