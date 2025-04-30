"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains supporting functions for model testing.
"""




# #############################################################################
# IMPORTS
import torch
import os
import sys

from models.model_classic import ModelClassic, constant_velocity_predictor, constant_acceleration_predictor
from models.model_bike_kinematics import ModelBikeKinematics
from models.model_ekf import ModelXKalman
from models.model_social_lstm import SocialLSTM
from models.model_social_bigat import SocialBiGAT
from models.model_physics_lstm import PhysicsLSTM
from models.model_gatsbi_v1 import GATSBIv1
from models.model_gatsbi_v2 import GATSBIv2
from models.model_gatsbi_v3 import GATSBIv3, load_gatsbi_modelv3
from models.model_gatsbi_v4 import GATSBIv4, load_gatsbi_modelv4

from models.model_utils import load_model, generate_model_scratch


# #############################################################################
# METHODS FOR TESTING
def load_model_testing(model_name, model_file_name, prediction_length, device, multimodal):
    model_path = "../data/4_models/"+model_file_name
    if os.path.exists(model_path):
        print("[model_loader.py] Use pretrained model from", model_path)
        if model_name=="social_lstm":
            model = load_model(SocialLSTM, model_path, device, prediction_length, multimodal)
        elif model_name=="social_bigat":
            model = load_model(SocialBiGAT, model_path, device, prediction_length, multimodal)
        elif model_name=="physics_lstm":
            model = load_model(PhysicsLSTM, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv1":
            model = load_model(GATSBIv1, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv2":
            model = load_model(GATSBIv2, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv3":
            if multimodal=="unimodal":  
                model = load_gatsbi_modelv3(model_path, device, prediction_length)
        elif model_name=="gatsbiv4":
            if multimodal=="unimodal":  
                model = load_gatsbi_modelv4(model_path, device, prediction_length)
        else:
            print("ERROR, model in ",model_file_name,"could not be found.")
            sys.exit(-1)
    else:
        if model_name=="const_v":
            model = ModelClassic(model_func=constant_velocity_predictor, prediction_length=prediction_length)
        elif model_name=="const_a":
            model = ModelClassic(model_func=constant_acceleration_predictor, prediction_length=prediction_length)
        elif model_name=="kinematics":
            model = ModelBikeKinematics(prediction_length=prediction_length)
        elif model_name=="xkalman":
            model = ModelXKalman(prediction_length=prediction_length)
        elif model_name=="social_lstm":
            model = generate_model_scratch(SocialLSTM, device, prediction_length, multimodal)
        elif model_name=="social_bigat":
            model = generate_model_scratch(SocialBiGAT, device, prediction_length, multimodal)
        elif model_name=="physics_lstm":
            model = generate_model_scratch(PhysicsLSTM, device, prediction_length, multimodal)
        elif model_name=="gatsbiv1":
            model = generate_model_scratch(GATSBIv1, device, prediction_length, multimodal)
        elif model_name=="gatsbiv2":
            model = generate_model_scratch(GATSBIv2, device, prediction_length, multimodal)
        elif model_name=="gatsbiv3":
            if multimodal=="unimodal":  
                model = GATSBIv3(prediction_length=prediction_length)
        elif model_name=="gatsbiv4":
            if multimodal=="unimodal":  
                model = GATSBIv4(prediction_length=prediction_length)
        else:
            print("ERROR failed to load model")
            sys.exit(-1)
    return model




# #############################################################################
# METHODS FOR TRAINING

def load_model_training(model_name, prediction_length, split, device, multimodal):
    # determine available models from last run
    files = os.listdir("../data/4_models/")
    relevant_files = [file for file in files if file.startswith(model_name+"_"+str(prediction_length)+"_"+split) and file.endswith(".model")]
    if multimodal=="multimodal_gmm" or multimodal=="multimodal_cvae":
        relevant_files = [file for file in files if file.startswith(model_name+"_"+str(prediction_length)+"_"+multimodal+"_"+split) and file.endswith(".model")]
    # generate model by loading from file
    last_epoch = -1
    if len(relevant_files)>0:
        most_recent_file = sorted(relevant_files)[-1]
        last_epoch = int(most_recent_file.split(".model")[0].split("_")[-1])
        model_path = "../data/4_models/"+most_recent_file
        print("[model_loader.py] Use pretrained model from", model_path)
        if model_name=="social_lstm":
            model = load_model(SocialLSTM, model_path, device, prediction_length, multimodal)
        elif model_name=="social_bigat":
            model = load_model(SocialBiGAT, model_path, device, prediction_length, multimodal)
        elif model_name=="physics_lstm":
            model = load_model(PhysicsLSTM, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv1":
            model = load_model(GATSBIv1, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv2":
            model = load_model(GATSBIv2, model_path, device, prediction_length, multimodal)
        elif model_name=="gatsbiv3":
            if multimodal=="unimodal":  
                model = load_gatsbi_modelv3(model_path, device, prediction_length)
        elif model_name=="gatsbiv4":
            if multimodal=="unimodal":  
                model = load_gatsbi_modelv4(model_path, device, prediction_length)
                
    # generate model by creating from scratch
    else:
        print("[model_loader.py] Create model from scatch")
        if model_name=="social_lstm":
            model = generate_model_scratch(SocialLSTM, device, prediction_length, multimodal)
        elif model_name=="social_bigat":
            model = generate_model_scratch(SocialBiGAT, device, prediction_length, multimodal)
        elif model_name=="physics_lstm":
            model = generate_model_scratch(PhysicsLSTM, device, prediction_length, multimodal)
        elif model_name=="gatsbiv1":
            model = generate_model_scratch(GATSBIv1, device, prediction_length, multimodal)
        elif model_name=="gatsbiv2":
            model = generate_model_scratch(GATSBIv2, device, prediction_length, multimodal)
        elif model_name=="gatsbiv3":
            if multimodal=="unimodal":  
                model = GATSBIv3(prediction_length=prediction_length)
        elif model_name=="gatsbiv4":
            if multimodal=="unimodal":  
                model = GATSBIv4(prediction_length=prediction_length)
        model.to(device)
    return model, last_epoch

def unpack_data(data, model_name, prediction_length):
    if model_name=="social_lstm":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
        neighbor_hists = data['neighbor_trajectory_history']
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, neighbor_hists)
    elif model_name=="social_bigat":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
        neighbor_hists = data['neighbor_trajectory_history']
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, neighbor_hists)
    elif model_name=="gatsbiv1" or model_name=="gatsbiv2" or model_name=="gatsbiv4":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
            # social feature        
        neighbor_hists = data['neighbor_trajectory_history']
        adj_matrixs = data["neighbor_adjacency_matrix"]
            # physics feature
        pred_cv = data["preds_cv"]
        pred_ca = data["preds_ca"]
        pred_bk = data["preds_bk"]
        pred_xk = data["preds_xk"]
            # road feature
        road_dist = data["ego_road_border_distance"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        pred_cv = pred_cv[:, :prediction_length, :]
        pred_ca = pred_ca[:, :prediction_length, :]
        pred_bk = pred_bk[:, :prediction_length, :]
        pred_xk = pred_xk[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, neighbor_hists, adj_matrixs, pred_cv, pred_ca, pred_bk, pred_xk, road_dist)
    elif model_name=="gatsbiv3":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
            # social feature        
        neighbor_hists = data['neighbor_trajectory_history']
        adj_matrixs = data["neighbor_adjacency_matrix"]
            # physics feature
        pred_cv = data["preds_cv"]
            # road feature
        road_dist = data["ego_road_border_distance"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        pred_cv = pred_cv[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, neighbor_hists, adj_matrixs, pred_cv, road_dist)
    elif model_name=="physics_lstm":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
        pred_cv = data["preds_cv"]
        pred_ca = data["preds_ca"]
        pred_bk = data["preds_bk"]
        pred_xk = data["preds_xk"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        pred_cv = pred_cv[:, :prediction_length, :]
        pred_ca = pred_ca[:, :prediction_length, :]
        pred_bk = pred_bk[:, :prediction_length, :]
        pred_xk = pred_xk[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, pred_cv, pred_ca, pred_bk, pred_xk)
    elif model_name=="const_v" or model_name=="const_a" or model_name=="kinematics" or model_name=="xkalman":
        # unpack data
        ego_hists = data['ego_trajectory_history']
        future_trajs = data['ego_trajectory_future']
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists)
    return dataset

def unpack_trajectory_prediction(model_results, model_name, multimodal):
    if model_name=="social_lstm":
        return model_results
    elif model_name=="social_bigat":
        return model_results
    elif model_name=="const_v":
        return model_results
    elif model_name=="const_a":
        return model_results
    elif model_name=="kinematics":
        return model_results
    elif model_name=="xkalman":
        return model_results
    elif model_name=="physics_lstm":
        return model_results
    elif model_name.startswith("gatsbi"):
        if multimodal=="unimodal":
            return model_results[0]
        else:
            return model_results