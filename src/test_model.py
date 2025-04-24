"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script generates data for different models for training and testing.
Usage: python test_model.py [1] [2] [3] [4]")
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi" or "const_v" or "const_a")
    [4] - model_file_name
    [5] - prediction_length in [s] (25, 50 , 75, 100)
    
Example:
    python test_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm social_lstm_25_5_0010.model 25
"""




# #############################################################################
# ### IMPORTS
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

from training.loss_functions import compute_ADE, compute_FDE
from models.model_loader import load_model_testing, unpack_testing_data, unpack_trajectory_prediction
import utils.constants as cs




# #############################################################################
# ### METHODS
def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python test_model.py [1] [2] [3] [4]")
    print(" [1] - relevant_video")
    print(" [2] - relevant_part")
    print(" [3] - model (\"social_lstm\" or \"gatsbi\" or \"const_v\" or \"const_a\")")
    print(" [4] - model_file_name")
    print(" [5] - prediction_length in [s] (25, 50 , 75, 100)")
    print("")
    print("Example: python test_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm social_lstm_25_5_0010.model 25")
    print("-------------------------------------------")

def load_testing_data(model_name, sequence):
    if model_name=="const_v" or model_name=="const_a":
        model_name = "social_lstm"
    data = torch.load('../data/3_testing_datasets/data_'+model_name+'_'+sequence+'.pt')
    return data

def test_model(model_name, model, testing_data, prediction_length, loss_functions):
    # prepare a dataloader
    dataset, future_trajs = unpack_testing_data(testing_data, model_name, prediction_length)
    # test model
    model_results = model(*dataset)
    pred_traj = unpack_trajectory_prediction(model_results, model_name)
    # evaluate model
    performances = {}
    for loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name]
        performances[loss_function_name] = loss_function(pred_traj, future_trajs)
    # return
    return performances

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
    model_file_name = run_arguments[4]
    prediction_length = int(run_arguments[5])
    
    # print info statement
    print("[test_model.py] Testing Model", relevant_video, relevant_part, model_name, model_file_name, prediction_length)
    
    # runargs check
    if not (model_name=="social_lstm" or model_name=="gatsbi" or model_name=="const_v" or model_name=="const_a"):
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
    if model_name=="const_a" or model_name=="const_v":
        model_file_name = "no"
        
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)
    
    # load testing data
    sequence = relevant_video + "-" + relevant_part
    testing_data = load_testing_data(model_name, sequence)

    # load model
    model = load_model_testing(model_name, model_file_name, prediction_length, device)
    
    # test model
    loss_functions = {"ADE": compute_ADE, "FDE": compute_FDE}
    performances = test_model(model_name, model, testing_data, prediction_length, loss_functions)
    
    # print result
    print(performances)

