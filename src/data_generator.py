"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script generates data for different models for training and testing.
Usage: python data_generator.py [1] [2] [3] [4]")
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi" or "physics_lstm")
    [4] - data_type ("train" or "test")
    
Example:
    python data_generator.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm train
"""




# #############################################################################
# ### IMPORTS
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import load_trajectories, get_unique_vehicles, get_frame_range
from data.data_gen_social_lstm import generate_data_social_lstm_all_batches
from data.data_gen_gatsbi import generate_data_gatsbi_all_batches
from data.data_gen_physics import generate_data_physics_all_batches
import utils.constants as cs




# #############################################################################
# ### METHODS

def generate_data_social_lstm(trajectory_data, data_type, batches):
    ego_hists, ego_pos, future_trajs, neighbor_hists, neighbor_pos = generate_data_social_lstm_all_batches(
        trajectory_data, batches, cs.HISTORY_LENGTH, cs.PREDICTION_LENGTH, cs.N_NEIGHBORS, data_type
    )
    data_dict = {
        'ego_trajectory_history': ego_hists,
        'ego_position': ego_pos,
        'ego_trajectory_future': future_trajs,
        'neighbor_trajectory_history': neighbor_hists,
        'neighbor_position': neighbor_pos
    }
    return data_dict

def generate_data_gatsbi(trajectory_data, data_type, batches):
    ego_hists, future_trajs, neighbor_hists, adj_matrixs, dists = generate_data_gatsbi_all_batches(
        trajectory_data, batches, cs.HISTORY_LENGTH, cs.PREDICTION_LENGTH, cs.N_NEIGHBORS, data_type
    )
    data_dict = {
        'ego_trajectory_history': ego_hists,
        'ego_trajectory_future': future_trajs,
        'neighbor_trajectory_history': neighbor_hists,
        'neighbor_adjacency_matrix': adj_matrixs,
        'ego_road_border_distance': dists
    }
    return data_dict

def generate_data_physics(trajectory_data, data_type, batches):
    ego_hists, future_trajs, preds_cv, preds_ca, preds_bk, preds_xk = generate_data_physics_all_batches(
        trajectory_data, batches, cs.HISTORY_LENGTH, cs.PREDICTION_LENGTH, cs.N_NEIGHBORS, data_type
    )
    data_dict = {
        'ego_trajectory_history': ego_hists,
        'ego_trajectory_future': future_trajs,
        'preds_cv': preds_cv,
        'preds_ca': preds_ca,
        'preds_bk': preds_bk,
        'preds_xk': preds_xk,
    }
    return data_dict

def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python data_generator.py [1] [2] [3] [4]")
    print(" [1] - relevant_video")
    print(" [2] - relevant_part")
    print(" [3] - model (\"social_lstm\" or \"gatsbi\" or \"physics_lstm\")")
    print(" [4] - data_type (\"train\" or \"test\")")
    print("")
    print("Example: python data_generator.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm train")
    print("-------------------------------------------")

def generate_batches(unique_vehicles, sequence, frame_from, frame_to):
    batches = []
    for vehicle_id in unique_vehicles:
        for frame in range(frame_from, frame_to):
            batches.append((sequence, vehicle_id, frame))
    return batches

def prepare_output_file_path(data_type, sequence, model):
    output_file_path = "../data/"
    if data_type=="train":
        output_file_path += "2_training_datasets/"
    else:
        output_file_path += "3_testing_datasets/"
    output_file_path += "data_"+model+"_"+sequence+".pt"
    return output_file_path

def generate_data(trajectory_data, batches, model, data_type):
    if model=="social_lstm":
        return generate_data_social_lstm(trajectory_data, data_type, batches)
    elif model=="gatsbi":
        return generate_data_gatsbi(trajectory_data, data_type, batches)
    elif model=="physics_lstm":
        return generate_data_physics(trajectory_data, data_type, batches)
    
def save_data(data_dict, output_file_path):
    torch.save(data_dict, output_file_path)

if __name__=="__main__":
    # parse runargs
    run_arguments = sys.argv
    if len(run_arguments)!=5:
        print("ERROR: invalid number of arguments")
        print_info()
        sys.exit(-1)
    relevant_video = run_arguments[1]
    relevant_part = run_arguments[2]
    model = run_arguments[3]
    data_type = run_arguments[4]
    
    # print info statement
    print("[data_generator.py] Generating Data For", relevant_video, relevant_part, model, data_type)
    
    # runargs check
    if not (model=="social_lstm" or model=="gatsbi" or model=="physics_lstm"):
        print("ERROR: invalid model")
        print_info()
        sys.exit(-1)
    if not (data_type=="train" or data_type=="test"):
        print("ERROR: invalid data_type")
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
    
    # generate data
        # load trajectory data
    trajectory_data = load_trajectories()
    sequence = relevant_video+"-"+relevant_part
    unique_vehicles = get_unique_vehicles(trajectory_data, sequence)
    frame_from, frame_to = get_frame_range(trajectory_data, sequence)
        # preparation
    batches = generate_batches(unique_vehicles, sequence, frame_from, frame_to)
    output_file_path = prepare_output_file_path(data_type, sequence, model)
        # generate data
    data_dict = generate_data(trajectory_data, batches, model, data_type)
        # save data
    save_data(data_dict, output_file_path)
