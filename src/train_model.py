"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script trains a model.
Usage: python train_model.py [1] [2] [3] [4] ([5])
    [1] - model ("social_lstm" or "social_bigat" or "gatsbi" or "physics_lstm")
    [2] - prediction_length in [s] (25, 50 , 75, 100)
    [3] - max_epochs
    [4] - split ("split_1" or "split_2" or "split_3" or "split_4" or "split_5")
    optional:
    [5] - multimodal ("unimodal" or "multimodal_gmm" or "multimodal_cvae")
    
Example:
    python train_model.py social_lstm 25 10 split_1 unimodal
"""




# #############################################################################
# ### IMPORTS
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

from training.testing_function import test_model
from models.model_loader import unpack_trajectory_prediction, load_model_training
from models.model_availability import ML_MODELS_UNIMODAL, ML_MODELS_MULTIMODAL_GMM, ML_MODELS_MULTIMODAL_CVAE
from data.dataset_loader import load_dataset
from training.loss_functions import compute_ADE_train, compute_FDE_train, gmm_loss
import utils.constants as cs




# #############################################################################
# ### METHODS
def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python train_model.py [1] [2 [3] [4]")
    print(" [1] - model (\"social_lstm\" or \"social_bigat\" or \"gatsbi\" or \"physics_lstm\")")
    print(" [2] - prediction_length in [s] (25, 50, 75, 100)")
    print(" [3] - max_epochs")
    print(" [4] - split (\"split_1\" or \"split_2\" or \"split_3\" or \"split_4\" or \"split_5\") ")
    print(" (optional):")
    print(" [5] - multi_modal (\"unimodal\" or \"multimodal_gmm\" or \"multimodal_cvae\")")
    print("")
    print("Example: python train_model.py social_lstm 25 50 split_1 unimodal")
    print("-------------------------------------------")




# #############################################################################
# ### MAIN LOGIC

if __name__=="__main__":
    # parse runargs
    run_arguments = sys.argv
    if len(run_arguments)<5:
        print("ERROR: invalid number of arguments")
        print_info()
        sys.exit(-1)
    model_name = run_arguments[1]
    prediction_length = int(run_arguments[2])
    max_epochs = int(run_arguments[3])
    split = run_arguments[4]
    multimodal = "unimodal"
    if len(run_arguments)==6:
        multimodal = run_arguments[5]
    
    # model_name="gatsbiv2"
    # prediction_length = 25
    # max_epochs = 3
    # split = "split_1"
    # multimodal = "multimodal_gmm"
    
    # print info statement
    print("[train_model.py] Training Model", model_name, prediction_length, max_epochs, split, multimodal)
    
    # runargs check
    if not (model_name=="social_lstm" or model_name=="social_bigat" or model_name.startswith("gatsbi") or model_name=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if (not split in cs.TRAIN_TEST_SPLITS):
        print("ERROR: invalid split")
        print_info()
        sys.exit(-1)
    if not (multimodal=="unimodal" or multimodal=="multimodal_gmm" or multimodal=="multimodal_cvae"):
        print("ERROR: invalid modality")
        print_info()
        sys.exit(-1)
    if multimodal=="unimodal":
        if not model_name in ML_MODELS_UNIMODAL:
            print("ERROR: modality not available for this model")
            print_info()
            sys.exit(-1)
    if multimodal=="multimodal_gmm":
        if not model_name in ML_MODELS_MULTIMODAL_GMM:
            print("ERROR: modality not available for this model")
            print_info()
            sys.exit(-1)
    if multimodal=="multimodal_cvae":
        if not model_name in ML_MODELS_MULTIMODAL_CVAE:
            print("ERROR: modality not available for this model")
            print_info()
            sys.exit(-1)
        
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)
    
    # prepare data
    training_dataset = load_dataset(model_name, cs.TRAIN_TEST_SPLITS[split]["TRAINING_VIDEOS"], prediction_length)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cs.BATCH_SIZE, shuffle=True)
    testing_dataset = load_dataset(model_name, cs.TRAIN_TEST_SPLITS[split]["TESTING_VIDEOS"], prediction_length)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=cs.BATCH_SIZE, shuffle=True)
    
    # load last available model
    model, last_epoch = load_model_training(model_name, prediction_length, split, device, multimodal)
    
    # define loss functions
    if multimodal=="unimodal":
        loss_function_training = compute_ADE_train
    else:
        loss_function_training = gmm_loss
    loss_functions_testing = {"ADE": compute_ADE_train, "FDE": compute_FDE_train}
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # train model
    print("[train_model.py] Everything prepared, lets start training on dataset, size:", [x.shape for x in training_dataset[0]])
    model.to(device)
    for epoch in range(last_epoch+1, max_epochs):
        # Conduct Training Over All Batches For One Epoch
        model.train()
        total_loss = 0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for batch in pbar:
            batch_data = [x.to(device) for x in batch]
            future_traj = batch_data[0]
            batch_feature_data = batch_data[1:]
            # Forward pass
            model_results = model(*batch_feature_data)
            model_res = unpack_trajectory_prediction(model_results, model_name)
            # Loss computation
            if multimodal=="unimodal":
                pred_traj = model_res
                loss = loss_function_training(pred_traj, future_traj)
            if multimodal=="multimodal_gmm":
                mu_x = model_res[0]
                mu_y = model_res[1]
                sigma_x = model_res[2]
                sigma_y = model_res[3]
                rho = model_res[4]
                pi = model_res[5]
                loss = loss_function_training(mu_x, mu_y, sigma_x, sigma_y, rho, pi, future_traj)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        # Determine Training Loss
        avg_loss = total_loss / num_batches
        print(f"[Epoch {epoch+1}] Average Training ADE Loss: {avg_loss:.4f}")        
        # Save Snapshot Of Model
        model_path = f"../data/4_models/{model_name}_{prediction_length:}_{split}_{epoch:02d}.model" # save model checkpoint after every epoch
        if multimodal=="multimodal_gmm" or multimodal=="multimodal_cvae":
            model_path = f"../data/4_models/{model_name}_{prediction_length:}_{multimodal}_{split}_{epoch:02d}.model" # save model checkpoint after every epoch
        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False) # downwards compatible saving
        # Determine Testing Loss
        performances = test_model(model_name, model, testing_loader, loss_functions_testing, prediction_length, device, multimodal)
        print(f"[Epoch {epoch+1}] Testing Loss: {performances}")       
        # Log Testing Loss
        f = open(model_path+"_perf.txt", "w+")
        f.write(str(performances))
        f.close()