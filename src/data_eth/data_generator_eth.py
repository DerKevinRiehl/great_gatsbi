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

import data_eth.trajectory_loader_eth
import data_eth.data_gen_social
import data_eth.data_gen_physics
import data_eth.data_gen_batch_info


# #############################################################################
# ### METHODS

def print_info():
    print("-------------------------------------------")
    print("Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs")
    print("-------------------------------------------")
    print("USAGE: python data_generator_eth.py")
    print("")
    print("Example: python data_generator_eth.py")
    print("-------------------------------------------")

def generate_batches(unique_vehicles, frame_from, frame_to):
    batches = []
    for vehicle_id in unique_vehicles:
        for frame in range(frame_from, frame_to):
            batches.append((vehicle_id, frame))
    return batches

def prepare_output_file_path(model_features, source):
    output_file_path = "../data/2_datasets/"
    output_file_path += "data_"+model_features+"_"+source+".pt"
    return output_file_path

def generate_data(trajectory_data, batches, model_features):
    if model_features=="social":
        return data_eth.data_gen_social.generate_data_social(trajectory_data, batches)
    elif model_features=="physics":
        return data_eth.data_gen_physics.generate_data_physics(trajectory_data, batches)
    elif model_features=="batches":
        return data_eth.data_gen_batch_info.generate_data_batches(trajectory_data, batches)
    
def save_data(data_dict, output_file_path):
    torch.save(data_dict, output_file_path)




# #############################################################################
# ### MAIN LOGIC

if __name__=="__main__":    
    source = "ETH"
    print("[data_generator_eth.py] Generating Data")
    for model_features in ["social"]:#["social", "physics", "batches"]:
        print("generating data", model_features)
        # generate data
            # load trajectory data
        if source=="ETH":
            trajectory_data = data_eth.trajectory_loader_eth.load_ETH()
        else:
            trajectory_data = data_eth.trajectory_loader_eth.load_HOTEL()
        unique_vehicles = data_eth.trajectory_loader_eth.get_unique_vehicles(trajectory_data)
        frame_from, frame_to = data_eth.trajectory_loader_eth.get_frame_range(trajectory_data)
            # preparation
        batches = generate_batches(unique_vehicles, frame_from, frame_to)
        output_file_path = prepare_output_file_path(model_features, source)
            # generate data
        data_dict = generate_data(trajectory_data, batches, model_features)
            # save data
        save_data(data_dict, output_file_path)
