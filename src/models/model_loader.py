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
from models.model_physics_lstm import PhysicsLSTM, load_physics_lstm_model
from models.model_gatsbi import GATSBI, load_gatsbi_model
from models.model_social_lstm import SocialLSTM, load_social_lstm_model



# #############################################################################
# METHODS FOR TESTING
def load_model_testing(model_name, model_file_name, prediction_length, device):
    model_path = "../data/4_models/"+model_file_name
    if os.path.exists(model_path):
        print("[model_loader.py] Use pretrained model from", model_path)
        if model_name=="social_lstm":
            model = load_social_lstm_model(model_path, device, prediction_length)
        elif model_name=="gatsbi":
            model = load_gatsbi_model(model_path, device, prediction_length)
        elif model_name=="physics_lstm":
            model = load_physics_lstm_model(model_path, device, prediction_length)
        else:
            print("ERROR, model in ",model_file_name,"could not be found.")
            sys.exit(-1)
    else:
        if model_name=="social_lstm":
            model = SocialLSTM(prediction_length=prediction_length)
        elif model_name=="gatsbi":
            model = GATSBI(prediction_length=prediction_length)
        elif model_name=="const_v":
            model = ModelClassic(model_func=constant_velocity_predictor, prediction_length=prediction_length)
        elif model_name=="const_a":
            model = ModelClassic(model_func=constant_acceleration_predictor, prediction_length=prediction_length)
        elif model_name=="kinematics":
            model = ModelBikeKinematics(prediction_length=prediction_length)
        elif model_name=="xkalman":
            model = ModelXKalman(prediction_length=prediction_length)
        elif model_name=="physics_lstm":
            model = PhysicsLSTM(prediction_length=prediction_length)
        else:
            print("ERROR failed to load model")
            sys.exit(-1)
    return model

def unpack_testing_data(testing_data, model_name, prediction_length):
    if model_name=="social_lstm":
        # unpack testing_data data
        ego_hists = testing_data['ego_trajectory_history']
        ego_pos = testing_data['ego_position']
        future_trajs = testing_data['ego_trajectory_future']
        neighbor_hists = testing_data['neighbor_trajectory_history']
        neighbor_pos = testing_data['neighbor_position']
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # create dataloader for batch processing
        return [ego_hists, ego_pos, neighbor_hists, neighbor_pos], future_trajs
    elif model_name=="gatsbi":
        # unpack testing_data data
        ego_hists = testing_data['ego_trajectory_history']
        future_trajs = testing_data['ego_trajectory_future']
        neighbor_hists = testing_data['neighbor_trajectory_history']
        adj_matrixs = testing_data["neighbor_adjacency_matrix"]
        road_dist = testing_data["ego_road_border_distance"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # Create DataLoader for batch processing
        return [ego_hists, neighbor_hists, adj_matrixs, road_dist], future_trajs
    elif model_name=="const_v" or model_name=="const_a" or model_name=="kinematics" or model_name=="xkalman":
        # unpack testing_data data
        ego_hists = testing_data['ego_trajectory_history']
        future_trajs = testing_data['ego_trajectory_future']
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        return [ego_hists], future_trajs
    elif model_name=="physics_lstm":
        # unpack training data
        ego_hists = testing_data['ego_trajectory_history']
        future_trajs = testing_data['ego_trajectory_future']
        pred_cv = testing_data["preds_cv"]
        pred_ca = testing_data["preds_ca"]
        pred_bk = testing_data["preds_bk"]
        pred_xk = testing_data["preds_xk"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        pred_cv = pred_cv[:, :prediction_length, :]
        pred_ca = pred_ca[:, :prediction_length, :]
        pred_bk = pred_bk[:, :prediction_length, :]
        pred_xk = pred_xk[:, :prediction_length, :]
        # create tensordataset
        return [ego_hists, pred_cv, pred_ca, pred_bk, pred_xk], future_trajs
    return None, None




# #############################################################################
# METHODS FOR TRAINING

def load_model_training(model_name, prediction_length, device):
    model_path = "../data/4_models/"+model_name+"_"+str(prediction_length)+"_"+str(5)+".model"
    if os.path.exists(model_path):
        print("[model_loader.py] Use pretrained model from", model_path)
        if model_name=="social_lstm":
            model = load_social_lstm_model(model_path, device, prediction_length)
        elif model_name=="gatsbi":
            model = load_gatsbi_model(model_path, device, prediction_length)
        elif model_name=="physics_lstm":
            model = load_physics_lstm_model(model_path, device, prediction_length)
    else:
        if model_name=="social_lstm":
            model = SocialLSTM(prediction_length=prediction_length)
        elif model_name=="gatsbi":
            model = GATSBI(prediction_length=prediction_length)
        elif model_name=="physics_lstm":
            model = PhysicsLSTM(prediction_length=prediction_length)
    return model, model_path

def unpack_training_data(training_data, model_name, batch_size, prediction_length):
    if model_name=="social_lstm":
        # unpack training data
        ego_hists = training_data['ego_trajectory_history']
        ego_pos = training_data['ego_position']
        future_trajs = training_data['ego_trajectory_future']
        neighbor_hists = training_data['neighbor_trajectory_history']
        neighbor_pos = training_data['neighbor_position']
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, ego_pos, neighbor_hists, neighbor_pos)
    elif model_name=="gatsbi":
        # unpack training data
        ego_hists = training_data['ego_trajectory_history']
        future_trajs = training_data['ego_trajectory_future']
        neighbor_hists = training_data['neighbor_trajectory_history']
        adj_matrixs = training_data["neighbor_adjacency_matrix"]
        road_dist = training_data["ego_road_border_distance"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, neighbor_hists, adj_matrixs, road_dist)
    elif model_name=="physics_lstm":
        # unpack training data
        ego_hists = training_data['ego_trajectory_history']
        future_trajs = training_data['ego_trajectory_future']
        pred_cv = training_data["preds_cv"]
        pred_ca = training_data["preds_ca"]
        pred_bk = training_data["preds_bk"]
        pred_xk = training_data["preds_xk"]
        # cut future trajectories to prediction length of model
        future_trajs = future_trajs[:, :prediction_length, :]
        pred_cv = pred_cv[:, :prediction_length, :]
        pred_ca = pred_ca[:, :prediction_length, :]
        pred_bk = pred_bk[:, :prediction_length, :]
        pred_xk = pred_xk[:, :prediction_length, :]
        # create tensordataset
        dataset = torch.utils.data.TensorDataset(future_trajs, ego_hists, pred_cv, pred_ca, pred_bk, pred_xk)
    return dataset

def unpack_trajectory_prediction(model_results, model_name):
    if model_name=="social_lstm":
        return model_results
    elif model_name=="gatsbi":
        return model_results[0]
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
