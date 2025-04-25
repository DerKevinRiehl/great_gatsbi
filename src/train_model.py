"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script trains a model.
Usage: python train_model.py [1] [2] [3] [4]")
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi" or "physics_lstm")
    [4] - prediction_length in [s] (25, 50 , 75, 100)
    [5] - n_epochs
    
Example:
    python train_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm 25 10
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

from models.model_loader import unpack_trajectory_prediction, unpack_training_data, load_model_training
from training.loss_functions import compute_ADE_train
import utils.constants as cs




# #############################################################################
# ### METHODS
def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python train_model.py [1] [2] [3] [4]")
    print(" [1] - relevant_video")
    print(" [2] - relevant_part")
    print(" [3] - model (\"social_lstm\" or \"gatsbi\" or \"physics_lstm\")")
    print(" [4] - prediction_length in [s] (25, 50 , 75, 100)")
    print(" [5] - n_epochs")
    print("")
    print("Example: python train_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm 25 10")
    print("-------------------------------------------")

def load_training_data(model_name, sequence):
    data = torch.load('../data/2_training_datasets/data_'+model_name+'_'+sequence+'.pt')
    return data

def train_model(model_name, model_path, model, training_data, prediction_length, batch_size, n_epochs, loss_function):
    # prepare a dataloader
    dataset = unpack_training_data(training_data, model_name, batch_size, prediction_length)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # train model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            batch_data = [x.to(device) for x in batch]
            future_traj = batch_data[0]
            batch_feature_data = batch_data[1:]
            # Forward pass
            model_results = model(*batch_feature_data)
            pred_traj = unpack_trajectory_prediction(model_results, model_name)
            # Loss computation
            loss = loss_function(pred_traj, future_traj)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / num_batches
        print(f"[Epoch {epoch+1}] Average Training ADE Loss: {avg_loss:.4f}")
        # Save model checkpoint after every epoch
        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False) # downwards compatible saving
    # return
    return model

if __name__=="__main__":
    # parse runargs
    run_arguments = sys.argv
    if len(run_arguments)!=6:
        print("ERROR: invalid number of arguments")
        print_info()
        sys.exit(-1)
    relevant_video = run_arguments[1]
    relevant_part = run_arguments[2]
    model_name = run_arguments[3]
    prediction_length = int(run_arguments[4])
    n_epochs = int(run_arguments[5])
    
    # print info statement
    print("[train_model.py] Training Model", relevant_video, relevant_part, model_name, prediction_length, n_epochs)
    
    # runargs check
    if not (model_name=="social_lstm" or model_name=="gatsbi" or model_name=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if not relevant_video in cs.VIDEOS:
        print("ERROR: invalid video")
        print_info()
        sys.exit(-1)
    if not relevant_part in cs.VIDEOS[relevant_video]:
        print("ERROR: invalid part")
        print_info()
        sys.exit(-1)
    
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)
    
    # load training data
    sequence = relevant_video + "-" + relevant_part
    training_data = load_training_data(model_name, sequence)
    
    # load model
    model, model_path = load_model_training(model_name, prediction_length, device)
    
    # train model
    loss_function = compute_ADE_train
    train_model(model_name, model_path, model, training_data, prediction_length, cs.BATCH_SIZE, n_epochs, loss_function)
    
    # save model
    model_path = "../data/4_models/"+model_name+"_"+str(prediction_length)+"_"+str(5)+"_"+sequence+".model"
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False) # downwards compatible saving

