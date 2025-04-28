"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script tests a model on all testing data.
Usage: python test_model.py [1] [2] [3]
    [1] - model ("social_lstm" or "social_bigat" or "gatsbi" or "const_v" or "const_a" or "kinematics" or "xkalman" or "physics_lstm")
    [2] - model_file_name
    [3] - prediction_length in [s] (25, 50, 75, 100)
    [4] - split ("split_1" or "split_2" or "split_3" or "split_4" or "split_5" or "all")
    optional:
    [5] - multimodal ("unimodal" or "multimodal_gmm" or "multimodal_cvae")
    
Example:
    python test_model_all.py social_lstm social_lstm_25_5_0010.model 25 split_1 unimodal
"""




# #############################################################################
# ### IMPORTS
import numpy as np
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

from training.testing_function import test_model
from training.loss_functions import compute_ADE_train, compute_FDE_train
from data.dataset_loader import load_dataset
from models.model_loader import load_model_testing
import utils.constants as cs




# #############################################################################
# ### METHODS
def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python test_model.py [1] [2] [3] [4]")
    print(" [1] - model (\"social_lstm\" or \"social_bigat\" or \"gatsbi\" or \"const_v\" or \"const_a\" or \"kinematics\" or \"xkalman\" or \"physics_lstm\")")
    print(" [2] - model_file_name")
    print(" [3] - prediction_length in [s] (25, 50, 75, 100)")
    print(" [4] - split (\"split_1\" or \"split_2\" or \"split_3\" or \"split_4\" or \"split_5\" or \"all\")")
    print(" (optional):")
    print(" [5] - multi_modal (\"unimodal\" or \"multimodal_gmm\" or \"multimodal_cvae\")")
    print("")
    print("Example: python test_model.py social_lstm social_lstm_25_5_0010.model 25")
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
    model_file_name = run_arguments[2]
    prediction_length = int(run_arguments[3])
    split = run_arguments[4]
    multimodal = "unimodal"
    if len(run_arguments)==6:
        multimodal = run_arguments[5]
        
    # prediction_length = 25*4
    # # model_name = "const_a"
    # model_name = "xkalman"
    # # model_name = "social_lstm"
    # model_file_name = "epochs_30/social_lstm_"+str(prediction_length)+"_5.model"
    # # model_name = "physics_lstm"
    # # model_file_name = "physics_lstm_25_5.model"
    # # model_name = "physics_lstm"
    # # model_file_name = "epochs_10_phlstm/physics_lstm_"+str(prediction_length)+"_5.model"
    # # model_name = "physics_lstm"
    # # model_file_name = "phlstm_32_10_v2/physics_lstm_"+str(prediction_length)+"_5.model"
    # split = "all"
    
    # runargs check
    if not (model_name=="social_lstm" or model_name=="social_bigat" or model_name.startswith("gatsbi") 
            or model_name=="const_v" or model_name=="const_a" or model_name=="kinematics"
            or model_name=="xkalman" or model_name=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if (not split in cs.TRAIN_TEST_SPLITS) and (not split=="all"):
        print("ERROR: invalid split")
        print_info()
        sys.exit(-1)
    if model_name=="const_a" or model_name=="const_v" or model_name=="kinematics" or model_name=="xkalman":
        model_file_name = "no"
    splits_to_test = list(cs.TRAIN_TEST_SPLITS.keys())
    if split!="all":
        splits_to_test = [split]
        
    # print info statement
    print("[test_model_all.py] Testing Model", model_name, model_file_name, prediction_length, split)
    
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[TORCH]\tRUNNING ON DEVICE:", device)
    
    # determine loss functions
    loss_functions = {"ADE": compute_ADE_train, "FDE": compute_FDE_train}
    
    # test relevant splits
    split_performances = []
    for split in splits_to_test:
        print("[test_model_all.py] Testing Model On Split", split)
    
        # load testing data
        testing_dataset = load_dataset(model_name, cs.TRAIN_TEST_SPLITS[split]["TESTING_VIDEOS"], prediction_length)
        testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=cs.BATCH_SIZE, shuffle=True)
    
        # load model
        model = load_model_testing(model_name, model_file_name, prediction_length, device, multimodal)
        
        # test model
        performances = test_model(model_name, model, testing_loader, loss_functions, prediction_length, device, multimodal)
        
        # print results
        print(">>Split Test Results [", model_name, model_file_name, prediction_length, "]")
        print(performances)            
        split_performances.append(performances)
        
    # aggregate test results avg and std
    print("===========================")
    print("Final Test Results")
    for loss in loss_functions:
        vals = []
        vals_a = []
        vals_b = []
        vals_c = []
        for entr in split_performances:
            if multimodal=="unimodal":
                vals.append(entr[loss])
            elif multimodal=="multimodal_gmm":
                vals_a.append(entr[loss][0])
                vals_b.append(entr[loss][1])
                vals_c.append(entr[loss][2])
        if multimodal=="unimodal":
            print(">>", loss, np.mean(vals), "[", np.std(vals), "]", "across", len(splits_to_test), "splits")
        elif multimodal=="multimodal_gmm":
            print(">>a", loss, np.mean(vals_a), "[", np.std(vals_a), "]", "across", len(splits_to_test), "splits")
            print(">>b", loss, np.mean(vals_b), "[", np.std(vals_b), "]", "across", len(splits_to_test), "splits")
            print(">>c", loss, np.mean(vals_c), "[", np.std(vals_c), "]", "across", len(splits_to_test), "splits")
            