"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for physics model.
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from models.model_classic import ModelClassic, constant_velocity_predictor, constant_acceleration_predictor
from models.model_bike_kinematics import ModelBikeKinematics
from models.model_ekf import ModelXKalman
from data.data_loader import get_relevant_neighbors, transform_ego_perspective
from data.data_loader import get_trajectory_history, get_trajectory_future
import utils.constants as cs




# #############################################################################
# METHODS
def generate_data_physics_all_batches(trajectory_data, batches, history_length, prediction_length, n_neighbors, data_type):
    lst_ego_hists = []
    lst_future_trajs = []
    lst_pred_cv = []
    lst_pred_ca = []
    lst_pred_bk = []
    lst_pred_xk = []
    # loop through all triplets
    for batch in tqdm(batches, desc="Loading Data"):
        ego_hist, predicted_traj, pred_cv_list, pred_ca_list, pred_bk_list, pred_xk_list = generate_training_data_physics_one_batch(
            trajectory_data, [batch],
            history_length=history_length,
            n_neighbors=n_neighbors,
            max_permutations=cs.BATCH_SIZE,
            data_type=data_type
        )
        if ego_hist is None:
            continue
        lst_ego_hists.extend(ego_hist)
        lst_future_trajs.extend(predicted_traj)
        lst_pred_cv.extend(pred_cv_list)
        lst_pred_ca.extend(pred_ca_list)
        lst_pred_bk.extend(pred_bk_list)
        lst_pred_xk.extend(pred_xk_list)
    # convert lists to tensors
    lst_ego_hists = torch.stack(lst_ego_hists)
    lst_future_trajs = torch.stack(lst_future_trajs)
    lst_pred_cv = torch.stack(lst_pred_cv)
    lst_pred_ca = torch.stack(lst_pred_ca)
    lst_pred_bk = torch.stack(lst_pred_bk)
    lst_pred_xk = torch.stack(lst_pred_xk)
    # return
    return lst_ego_hists, lst_future_trajs, lst_pred_cv, lst_pred_ca, lst_pred_bk, lst_pred_xk

def generate_training_data_physics_one_batch(trajectory_data, batch_triplets, history_length=100, n_neighbors=5, max_permutations=32, data_type="train"):
    ego_hists = []
    predicted_trajs = []
    pred_cvs = []
    pred_cas = []
    pred_bkins = []
    pred_xkals = []
    for sequence, ego_vehicle_id, frame_id in batch_triplets:
        if not data_type=="train":
            max_permutations = 1
        for n_repetitions in range(0, max_permutations):
            if n_repetitions==0:
                lane_xy, pred_traj, pred_cv, pred_ca, pred_bkin, pred_xkal = generate_training_data_physics_one_record(trajectory_data, n_neighbors, history_length, sequence, ego_vehicle_id, frame_id, data_type, skip_regen=False)
                last_pred_cv = pred_cv
                last_pred_ca = pred_ca
                last_pred_bkin = pred_bkin
                last_pred_xkal = pred_xkal
            else:
                lane_xy, pred_traj, pred_cv, pred_ca, pred_bkin, pred_xkal = generate_training_data_physics_one_record(trajectory_data, n_neighbors, history_length, sequence, ego_vehicle_id, frame_id, data_type, skip_regen=True)
                pred_cv = last_pred_cv
                pred_ca = last_pred_ca
                pred_bkin = last_pred_bkin
                pred_xkal = last_pred_xkal
            if lane_xy is None:
                continue
            ego_hists.append(lane_xy)
            predicted_trajs.append(pred_traj)
            pred_cvs.append(pred_cv)
            pred_cas.append(pred_ca)
            pred_bkins.append(pred_bkin)
            pred_xkals.append(pred_xkal)
    # conversion to tensors
    if len(ego_hists)==0:
        return None, None, None, None, None, None
    ego_hist = torch.tensor(np.stack(ego_hists, axis=0), dtype=torch.float32) 
    predicted_traj = torch.tensor(np.stack(predicted_trajs, axis=0), dtype=torch.float32)  
    pred_cv_list = torch.tensor(np.stack(pred_cvs, axis=0), dtype=torch.float32) 
    pred_ca_list = torch.tensor(np.stack(pred_cas, axis=0), dtype=torch.float32) 
    pred_bk_list = torch.tensor(np.stack(pred_bkins, axis=0), dtype=torch.float32) 
    pred_xk_list = torch.tensor(np.stack(pred_xkals, axis=0), dtype=torch.float32) 
    # Now you have:
    # ego_hist: (batch_size, history_length, 2)
    # predicted_traj: (batch_size, prediction_length, 2)
    # pred_cv_list: (batch_size, history_length, 2)
    # pred_ca_list: (batch_size, history_length, 2)
    # pred_bk_list: (batch_size, history_length, 2)
    # pred_xk_list: (batch_size, history_length, 2)
    return ego_hist, predicted_traj, pred_cv_list, pred_ca_list, pred_bk_list, pred_xk_list

def generate_training_data_physics_one_record(trajectory_data, n_neighbors, history_length, sequence, ego_vehicle_id, frame_id, data_type="train", skip_regen=False):
    prediction_length = 100
    # find neighbors for this record
    relevant_neighbors = get_relevant_neighbors(trajectory_data, sequence, frame_id, ego_vehicle_id, n_neighbors)
    if data_type=="train":
        random.shuffle(relevant_neighbors)
    df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
    if df_trajectory is None:
        return None, None, None, None, None, None
    df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, history_length)
    df_veh_future = get_trajectory_future(df_trajectory, ego_vehicle_id, frame_id, prediction_length)
    # ego history
    ego_hist = df_veh_history[["Lane_X", "Lane_Y"]].to_numpy()
    if ego_hist.shape[0] < history_length:
        pad_len = history_length - ego_hist.shape[0]
        ego_hist = np.pad(ego_hist, ((pad_len, 0), (0, 0)), mode='constant')
    else:
        ego_hist = ego_hist[-history_length:]
    # future trajectory
    pred_traj = df_veh_future[["Lane_X", "Lane_Y"]].to_numpy()
    if pred_traj.shape[0] < prediction_length:
        pad_len = prediction_length - pred_traj.shape[0]
        pred_traj = np.pad(pred_traj, ((0, pad_len), (0, 0)), mode='constant')
    else:
        pred_traj = pred_traj[:prediction_length]
    if skip_regen:
        pred_cv = None
        pred_ca = None
        pred_kin = None
        pred_xkal = None
    else:
        # const_v model future trajectory
        model = ModelClassic(model_func=constant_velocity_predictor, prediction_length=prediction_length)
        pred_cv = model([ego_hist])
        pred_cv = pred_cv[0]
        pred_cv = np.asarray(pred_cv)
        # const_a model future trajectory
        model = ModelClassic(model_func=constant_acceleration_predictor, prediction_length=prediction_length)
        pred_ca = model([ego_hist])
        pred_ca = pred_ca[0]
        pred_ca = np.asarray(pred_ca)
        # physics kinematics model future trajectory
        model = ModelBikeKinematics(prediction_length=prediction_length)
        pred_kin = model([ego_hist])
        pred_kin = pred_kin[0]
        pred_kin = np.asarray(pred_kin)
        # xkalman model future trajectory
        model = ModelXKalman(prediction_length=prediction_length)
        train_ego_hist = np.expand_dims(ego_hist, axis=0)
        pred_xkal = model(train_ego_hist)
        pred_xkal = pred_xkal[0]
        pred_xkal = np.asarray(pred_xkal)
    # return
    return ego_hist, pred_traj, pred_cv, pred_ca, pred_kin, pred_xkal