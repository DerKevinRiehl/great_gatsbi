"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script tests a model on all testing data.
Usage: python test_model.py [1] [2] [3] [4]")
    [1] - model ("social_lstm" or "gatsbi" or "const_v" or "const_a" or "kinematics" or "xkalman" or "physics_lstm")
    [2] - model_file_name
    [3] - prediction_length in [s] (25, 50, 75, 100)
    
Example:
    python test_model_all.py social_lstm social_lstm_25_5_0010.model 25
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
    print(" [1] - model (\"social_lstm\" or \"gatsbi\" or \"const_v\" or \"const_a\" or \"kinematics\" or \"xkalman\" or \"physics_lstm\")")
    print(" [2] - model_file_name")
    print(" [3] - prediction_length in [s] (25, 50 , 75, 100)")
    print("")
    print("Example: python test_model.py social_lstm social_lstm_25_5_0010.model 25")
    print("-------------------------------------------")

def load_testing_data(model_name, sequence):
    if model_name=="const_v" or model_name=="const_a" or model_name=="kinematics" or model_name=="xkalman":
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
    # # parse runargs
    # run_arguments = sys.argv
    # if len(run_arguments)!=4:
    #     print("ERROR: invalid number of arguments")
    #     print_info()
    #     sys.exit(-1)
    # model_name = run_arguments[1]
    # model_file_name = run_arguments[2]
    # prediction_length = int(run_arguments[3])
    
    prediction_length = 25*1
    # model_name = "const_v"
    # model_name = "xkalman"
    # model_name = "social_lstm"
    # model_file_name = "epochs_30/social_lstm_"+str(prediction_length)+"_5.model"
    # model_name = "physics_lstm"
    # model_file_name = "physics_lstm_25_5.model"
    # model_name = "physics_lstm"
    # model_file_name = "epochs_10_phlstm/physics_lstm_"+str(prediction_length)+"_5.model"
    model_name = "physics_lstm"
    model_file_name = "phlstm_32_5_v2/physics_lstm_"+str(prediction_length)+"_5.model"

    # print info statement
    print("[test_model_all.py] Testing Model", model_name, model_file_name, prediction_length)
    
    # runargs check
    if not (model_name=="social_lstm" or model_name=="gatsbi" 
            or model_name=="const_v" or model_name=="const_a" or model_name=="kinematics"
            or model_name=="xkalman" or model_name=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if model_name=="const_a" or model_name=="const_v" or model_name=="kinematics" or model_name=="xkalman":
        model_file_name = "no"
        
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)

    # load model
    model = load_model_testing(model_name, model_file_name, prediction_length, device)
    
    results = []
    for relevant_video in cs.TESTING_VIDEOS:
        for relevant_part in cs.VIDEOS[relevant_video]:
            # load testing data
            sequence = relevant_video + "-" + relevant_part
            testing_data = load_testing_data(model_name, sequence)
            duration = cs.VIDEOS[relevant_video][relevant_part][1] - cs.VIDEOS[relevant_video][relevant_part][0]
        
            # test model
            loss_functions = {"ADE": compute_ADE, "FDE": compute_FDE}
            performances = test_model(model_name, model, testing_data, prediction_length, loss_functions)
            
            # print result
            print(sequence, duration, performances)            
            results.append([sequence, duration, performances])

    complete_results = {}
    total_duration = 0
    for key in list(results[0][2].keys()):
        complete_results[key] = 0
    for run in results:
        duration = run[1]
        total_duration += duration
        res = run[2]
        for key in list(results[0][2].keys()):
            complete_results[key] += res[key]*duration
    final_results = {}
    for key in list(results[0][2].keys()):
        final_results[key] = complete_results[key]/total_duration
    print(">>Final Results [", model_name, model_file_name, prediction_length, "]")
    print(final_results)