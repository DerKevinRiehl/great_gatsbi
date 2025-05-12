"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This runnable Python script generates featured data for different models for training and testing.
"""




# #############################################################################
# ### IMPORTS
import torch
import warnings
warnings.filterwarnings("ignore")

from data.trajectory_loader import load_trajectories, get_unique_vehicles, get_frame_range
from data.data_gen_road import generate_data_road
from data.data_gen_social import generate_data_social
from data.data_gen_physics import generate_data_physics
from data.data_gen_batch_info import generate_data_batches
import utils.constants as cs




# #############################################################################
# ### METHODS

def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism")
    print("-------------------------------------------")
    print("USAGE: python data_generator.py")
    print("")
    print("Example: python data_generator.py")
    print("-------------------------------------------")

def generate_batches(unique_vehicles, sequence, frame_from, frame_to):
    batches = []
    for vehicle_id in unique_vehicles:
        for frame in range(frame_from, frame_to):
            batches.append((sequence, vehicle_id, frame))
    return batches

def prepare_output_file_path(sequence, model_features):
    output_file_path = "../data/2_datasets/"
    output_file_path += "data_"+model_features+"_"+sequence+".pt"
    return output_file_path

def generate_data(trajectory_data, batches, model_features):
    if model_features=="road":
        return generate_data_road(trajectory_data, batches)
    elif model_features=="social":
        return generate_data_social(trajectory_data, batches)
    elif model_features=="physics":
        return generate_data_physics(trajectory_data, batches)
    elif model_features=="batches":
        return generate_data_batches(trajectory_data, batches)
    
def save_data(data_dict, output_file_path):
    torch.save(data_dict, output_file_path)




# #############################################################################
# ### MAIN LOGIC

if __name__=="__main__":    
    print("[data_generator.py] Generating Data")
    for model_features in ["road", "social", "physics", "batches"]:
        for relevant_video in cs.VIDEOS_PARTS:
            for relevant_part in cs.VIDEOS_PARTS[relevant_video]:
                print("generating data", model_features, relevant_video, relevant_part)
                # generate data
                    # load trajectory data
                trajectory_data = load_trajectories()
                sequence = relevant_video+"-"+relevant_part
                unique_vehicles = get_unique_vehicles(trajectory_data, sequence)
                frame_from, frame_to = get_frame_range(trajectory_data, sequence)
                    # preparation
                batches = generate_batches(unique_vehicles, sequence, frame_from, frame_to)
                output_file_path = prepare_output_file_path(sequence, model_features)
                    # generate data
                data_dict = generate_data(trajectory_data, batches, model_features)
                    # save data
                save_data(data_dict, output_file_path)
