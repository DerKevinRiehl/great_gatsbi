"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for social_lstm model.
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import get_relevant_neighbors, transform_ego_perspective
from data.data_loader import get_trajectory_history, get_trajectory_future
import utils.constants as cs




# #############################################################################
# METHODS
def generate_data_social_lstm_all_batches(trajectory_data, batches, history_length, prediction_length, n_neighbors, data_type):
    ego_hists = []
    ego_pos = []
    future_trajs = []
    neighbor_hists = []
    neighbor_pos = []
    # loop through all triplets
    for batch in tqdm(batches, desc="Loading Data"):
        ego_hist, ego_pos_data, future_traj, neighbor_hists_data, neighbor_pos_data = generate_training_data_social_lstm_one_batch(
            trajectory_data, [batch],
            history_length=history_length,
            prediction_length=prediction_length,
            n_neighbors=n_neighbors,
            max_permutations=cs.BATCH_SIZE,
            data_type=data_type
        )
        if ego_hist is None:
            continue
        ego_hists.extend(ego_hist)
        ego_pos.extend(ego_pos_data)
        future_trajs.extend(future_traj)
        neighbor_hists.extend(neighbor_hists_data)
        neighbor_pos.extend(neighbor_pos_data)
    # convert lists to tensors
    ego_hists = torch.stack(ego_hists)
    ego_pos = torch.stack(ego_pos)
    future_trajs = torch.stack(future_trajs)
    neighbor_hists = torch.stack(neighbor_hists)
    neighbor_pos = torch.stack(neighbor_pos)
    # return
    return ego_hists, ego_pos, future_trajs, neighbor_hists, neighbor_pos

def generate_training_data_social_lstm_one_batch(trajectory_data, batch_triplets, history_length=100, prediction_length=25, n_neighbors=5, max_permutations=32, data_type="train"):
    ego_hists = []
    ego_positions = []
    predicted_trajs = []
    neighbor_hists_list = []
    neighbor_positions = []
    for sequence, ego_vehicle_id, frame_id in batch_triplets:
        neighbor_trajs, lane_xy, pred_traj = generate_training_data_social_lstm_one_record(trajectory_data, n_neighbors, history_length, prediction_length, sequence, ego_vehicle_id, frame_id)
        if neighbor_trajs is None:
            continue
        if not data_type=="train":
            max_permutations = 1
        # permute data (vary order of neighbors for data imputation)
        all_perms = list(permutations(range(n_neighbors)))
        if max_permutations is not None:
            np.random.shuffle(all_perms)
            all_perms = all_perms[:max_permutations]
        for perm in all_perms:
            permuted_trajs = [neighbor_trajs[i] for i in perm]
            permuted_positions = [traj[-1] for traj in permuted_trajs]
            ego_hists.append(lane_xy)
            ego_positions.append(lane_xy[-1])
            predicted_trajs.append(pred_traj)
            neighbor_hists_list.append(np.stack(permuted_trajs, axis=0))
            neighbor_positions.append(np.stack(permuted_positions, axis=0))
    # conversion to tensors
    if len(ego_hists)==0:
        return None, None, None, None, None
    ego_hist = torch.tensor(np.stack(ego_hists, axis=0), dtype=torch.float32)
    ego_pos = torch.tensor(np.stack(ego_positions, axis=0), dtype=torch.float32)
    predicted_traj = torch.tensor(np.stack(predicted_trajs, axis=0), dtype=torch.float32)
    neighbor_hists = torch.tensor(np.stack(neighbor_hists_list, axis=0), dtype=torch.float32)  
    neighbor_pos = torch.tensor(np.stack(neighbor_positions, axis=0), dtype=torch.float32)
    # resulting dimensions:
    # ego_hist: (batch_size, history_length, 2)
    # ego_pos: (batch_size, 2)
    # predicted_traj: (batch_size, prediction_length, 2)
    # neighbor_hists: (batch_size, n_neighbors, history_length, 2)
    # neighbor_pos: (batch_size, n_neighbors, 2)
    return ego_hist, ego_pos, predicted_traj, neighbor_hists, neighbor_pos

def generate_training_data_social_lstm_one_record(trajectory_data, n_neighbors, history_length, prediction_length, sequence, ego_vehicle_id, frame_id):
    # find neighbors for this record
    relevant_neighbors = get_relevant_neighbors(trajectory_data, sequence, frame_id, ego_vehicle_id, n_neighbors=n_neighbors)
    df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
    if df_trajectory is None:
        return None, None, None
    df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, history_length)
    df_veh_future = get_trajectory_future(df_trajectory, ego_vehicle_id, frame_id, prediction_length)
    # ego history and position
    lane_xy = df_veh_history[["Lane_X", "Lane_Y"]].to_numpy()
    if lane_xy.shape[0] < history_length:
        pad_len = history_length - lane_xy.shape[0]
        lane_xy = np.pad(lane_xy, ((pad_len, 0), (0, 0)), mode='constant')
    else:
        lane_xy = lane_xy[-history_length:]
    # future trajectory
    pred_traj = df_veh_future[["Lane_X", "Lane_Y"]].to_numpy()
    if pred_traj.shape[0] < prediction_length:
        pad_len = prediction_length - pred_traj.shape[0]
        pred_traj = np.pad(pred_traj, ((0, pad_len), (0, 0)), mode='constant')
    else:
        pred_traj = pred_traj[:prediction_length]
    # neighbor histories
    neighbor_trajs = []
    for neighbor_id in relevant_neighbors:
        df = get_trajectory_history(df_trajectory, neighbor_id, frame_id, history_length)
        arr = df[["Lane_X", "Lane_Y"]].to_numpy()
        if arr.shape[0] < history_length:
            pad_len = history_length - arr.shape[0]
            arr = np.pad(arr, ((pad_len, 0), (0, 0)), mode='constant')
        else:
            arr = arr[-history_length:]
        neighbor_trajs.append(arr)
    while len(neighbor_trajs) < n_neighbors:
        neighbor_trajs.append(np.zeros((history_length, 2), dtype=np.float32))
    neighbor_trajs = neighbor_trajs[:n_neighbors]
    # return
    return neighbor_trajs, lane_xy, pred_traj