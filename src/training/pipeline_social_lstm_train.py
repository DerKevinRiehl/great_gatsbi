"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This script is to train a model based on training data.
"""




# #############################################################################
# IMPORTS
import torch
import torch.optim as optim
from tqdm import tqdm
import os

import utils.constants as cs
from models.model_social_lstm import SocialLSTM, load_social_lstm_model




# #############################################################################
# TORCH SETUP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON DEVICE:", device)






# #############################################################################

# DEFINE NETWORK PARAMETERS    
batch_size = 32
prediction_length = 25*1
history_length = 100 
n_neighbours = 5


import sys
arguments = sys.argv
prediction_length = int(arguments[1])
SEQUENCE_VID = arguments[2]

# SEQUENCE_VID = "DJI_20240906103036_0003_D.MP4"

MODEL_PATH_TEMP = "social_lstm_"+str(prediction_length)+"_"+str(n_neighbours)+".model"

# SEQUENCE_VID = "DJI_20240906103036_0003_D.MP4"
# PARTS = ["PART_1", "PART_2", "PART_3", "PART_4"]

# SEQUENCE_VID = "DJI_20240906103442_0004_D.MP4"
# PARTS = ["PART_1", "PART_2"]

# SEQUENCE_VID = "DJI_20240906103850_0005_D.MP4"
# PARTS = ["PART_1"]

# SEQUENCE_VID = "DJI_20240906105321_0009_D.MP4"
# PARTS = ["PART_1"]

# SEQUENCE_VID = "DJI_20240906105621_0010_D.MP4"
# PARTS = ["PART_1", "PART_2", "PART_3", "PART_4", "PART_5", "PART_6"]

# SEQUENCE_VID = "DJI_20240906110027_0011_D.MP4"
# PARTS = ["PART_1", "PART_2", "PART_3", "PART_4", "PART_5"]

# SEQUENCE_VID = "DJI_20240906103036_0003_D.MP4"
# # PARTS = ["PART_1", "PART_2", "PART_3", "PART_4"]
# PARTS = ["PART_3"]

PARTS = list(REL_FRAMES[SEQUENCE_VID].keys())



# #############################################################################
# DO TRAINING
for PART in PARTS:
    SEQUENCE = SEQUENCE_VID+"-"+PART
    MODEL_PATH = "social_lstm_"+str(prediction_length)+"_"+str(n_neighbours)+"_"+SEQUENCE_VID.split(".MP4")[0].split("_")[-2]+".model"
    print(">>", SEQUENCE, prediction_length)
    
    # LOAD DATA
    data = torch.load('training_data_'+SEQUENCE+'.pt')
    ego_hists = data['ego_hists']
    ego_pos = data['ego_pos']
    future_trajs = data['future_trajs']
    neighbor_hists = data['neighbor_hists']
    neighbor_pos = data['neighbor_pos']
    
    # CUT FUTURE TRAJS DEPENDING ON PREDITION LENGTH
    future_trajs = future_trajs[:, :prediction_length, :]

    # Create DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(ego_hists, ego_pos, future_trajs, neighbor_hists, neighbor_pos)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # LOAD MODEL IF ALREADY EXIST
    if os.path.exists(MODEL_PATH_TEMP):
        model = load_social_lstm_model(SocialLSTM, MODEL_PATH_TEMP, device, prediction_length)
    else:
        model = SocialLSTM(prediction_length=prediction_length)
        
    # TRAIN MODEL
    def train_social_lstm_model(model, history_length, prediction_length,
                                n_neighbours, device, epochs=n_epochs):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.to(device)
    
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
    
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
            for batch in pbar:
                ego_hist, ego_pos, future_traj, neighbor_hists, neighbor_pos = [x.to(device) for x in batch]
    
                # Forward pass
                pred_traj = model(ego_hist, ego_pos, neighbor_hists, neighbor_pos)
    
                # Loss computation
                loss = compute_ADE(pred_traj, future_traj)
    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
    
            avg_loss = total_loss / num_batches
            print(f"[Epoch {epoch+1}] Average Training ADE Loss: {avg_loss:.4f}")
    
            # Save model checkpoint
            torch.save(model.state_dict(), MODEL_PATH_TEMP, _use_new_zipfile_serialization=False) # downwards compatible saving
    
    train_social_lstm_model(model, 
                            history_length, 
                            prediction_length,
                            n_neighbours, 
                            device=device, 
                            epochs=n_epochs)
    
    torch.save(model.state_dict(), MODEL_PATH, _use_new_zipfile_serialization=False) # downwards compatible saving