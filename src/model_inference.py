"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script generates inference data from models for visualization purposes as CSV file.
Usage: python model_inference.py [1] [2] [3]
    [1] - model ("social_lstm" or "gatsbi" or "const_v" or "const_a" or "kinematics" or "xkalman" or "physics_lstm")
    [2] - model_file_name
    [3] - sequence
    [4] - prediction_length in [s] (25, 50, 75, 100)
    [5] - output_file_name
    
Example:
    python model_inference.py physics_lstm physics_lstm_100_08.model DJI_20240906103850_0005_D.MP4-PART_1 25 inference.txt
"""




# #############################################################################
# ### IMPORTS
import torch
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models.model_loader import unpack_trajectory_prediction
from data.dataset_loader import load_dataset_inference
from models.model_loader import load_model_testing
import utils.constants as cs




# #############################################################################
# ### METHODS
def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python test_model.py [1] [2] [3] [4]")
    print(" [1] - model (\"social_lstm\" or \"gatsbi\" or \"const_v\" or \"const_a\" or \"kinematics\" or \"xkalman\" or \"physics_lstm\")")
    print(" [2] - model_file_name")
    print(" [3] - sequence")
    print(" [4] - prediction_length in [s] (25, 50 , 75, 100)")
    print(" [5] - output_file_name")
    
    print("")
    print("Example: python test_model.py physics_lstm physics_lstm_100_08.model DJI_20240906103850_0005_D.MP4-PART_1 25 inference.txt")
    print("-------------------------------------------")


def model_inference(model_name, model, test_loader, prediction_length, device):
    model.eval()
    all_pred_trajs = []
    all_future_trajs = []
    attentions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            batch_data = [x.to(device) for x in batch]
            future_traj = batch_data[0][:, :prediction_length, :]
            batch_feature_data = batch_data[1:]
            # Forward pass
            model_results = model(*batch_feature_data)
            if model_name.startswith("gatsbi"):
                res = model_results
                pred_traj = res[0]
                attention = res[1] 
                attentions.append(attention)
            else:
                pred_traj = unpack_trajectory_prediction(model_results, model_name)
            all_pred_trajs.append(pred_traj)
            all_future_trajs.append(future_traj)
    # Concatenate all predictions and targets
    all_pred_trajs = torch.cat(all_pred_trajs, dim=0)
    all_future_trajs = torch.cat(all_future_trajs, dim=0)
    if len(attentions)>0:
        attentions = torch.cat(attentions, dim=0)
    return all_pred_trajs, all_future_trajs, attentions


# #############################################################################
# ### MAIN LOGIC

if __name__=="__main__":
    # parse runargs
    run_arguments = sys.argv
    if len(run_arguments)!=6:
        print("ERROR: invalid number of arguments")
        print_info()
        sys.exit(-1)
    model_name = run_arguments[1]
    model_file_name = run_arguments[2]
    sequence = run_arguments[3]
    prediction_length = int(run_arguments[4])
    output_file_name = run_arguments[5]
    
    # model_name = "gatsbiv1"
    # model_file_name = "gatsbiv1_100_03.model"
    # # model_name = "physics_lstm"
    # # model_file_name = "physics_lstm_100_08.model"
    # sequence = "DJI_20240906103850_0005_D.MP4-PART_1"
    # # sequence = "DJI_20240906105321_0009_D.MP4-PART_1"
    # prediction_length = int(100)
    # output_file_name = "inference_test.txt"
    
    # runargs check
    if not (model_name=="social_lstm" or model_name.startswith("gatsbi") 
            or model_name=="const_v" or model_name=="const_a" or model_name=="kinematics"
            or model_name=="xkalman" or model_name=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if model_name=="const_a" or model_name=="const_v" or model_name=="kinematics" or model_name=="xkalman":
        model_file_name = "no"
        
    # print info statement
    print("[test_model_all.py] Model Inference", model_name, model_file_name, sequence, prediction_length, output_file_name)
    
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)
    
    # load testing data
    batches, testing_dataset = load_dataset_inference(model_name, sequence, prediction_length)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=cs.BATCH_SIZE, shuffle=False)
    
    # load model
    model = load_model_testing(model_name, model_file_name, prediction_length, device)
        
    # test model
    all_pred_trajs, all_future_trajs, attentions = model_inference(model_name, model, testing_loader, prediction_length, device)
    if attentions != []:
        attentions = attentions[:, 5, :] # from ego to neighbors
    
    # store results
    fW = open(output_file_name, "w+")
    fW.write("sequence\tbicycle\tframe_id\tfuture_coords_x\tfuture_coords_y\tpred_coords_x\tpred_coords_y")
    if model_name.startswith("gatsbi"):
        fW.write("\tattention_weights")
    fW.write("\n")
    for idx in range(0, len(batches)):
        fW.write(str(batches[idx][0]))
        fW.write("\t")
        fW.write(str(batches[idx][1]))
        fW.write("\t")
        fW.write(str(batches[idx][2]))
        fW.write("\t")
        fW.write(str(all_future_trajs[idx,:,0].tolist()))
        fW.write("\t")
        fW.write(str(all_future_trajs[idx,:,1].tolist()))
        fW.write("\t")
        fW.write(str(all_pred_trajs[idx,:,0].tolist()))
        fW.write("\t")
        fW.write(str(all_pred_trajs[idx,:,1].tolist()))
        if model_name.startswith("gatsbi"):
            fW.write("\t")
            fW.write(str(attentions[0].tolist()))
        fW.write("\n")
    fW.close()