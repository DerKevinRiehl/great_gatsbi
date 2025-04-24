"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for gatsbi model.
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import get_relevant_neighbors, transform_ego_perspective
from data.data_loader import get_trajectory_history, get_trajectory_future
import utils.constants as cs




# #############################################################################
# METHODS
def generate_data_gatsbi_all_batches(trajectory_data, batches, history_length, prediction_length, n_neighbors, data_type):
    lst_ego_hists = []
    lst_future_trajs = []
    lst_neighbor_hists = []
    lst_adj_matrix = []
    lst_dist = []
    # loop through all triplets
    for batch in tqdm(batches, desc="Loading Data"):
        ego_hist, predicted_traj, neighbor_hists, adj_matrixs, dist = generate_training_data_gatsbi_one_batch(
            trajectory_data, [batch],
            history_length=history_length,
            prediction_length=prediction_length,
            n_neighbors=n_neighbors,
            max_permutations=cs.BATCH_SIZE,
            data_type=data_type
        )
        if ego_hist is None:
            continue
        lst_ego_hists.extend(ego_hist)
        lst_future_trajs.extend(predicted_traj)
        lst_neighbor_hists.extend(neighbor_hists)
        lst_adj_matrix.extend(adj_matrixs)
        lst_dist.extend(dist)
    # convert lists to tensors
    lst_ego_hists = torch.stack(lst_ego_hists)
    lst_future_trajs = torch.stack(lst_future_trajs)
    lst_neighbor_hists = torch.stack(lst_neighbor_hists)
    lst_adj_matrix = torch.stack(lst_adj_matrix)
    lst_dist = torch.stack(lst_dist)
    print(">>>>YYY")
    print(lst_ego_hists.shape)
    print(lst_future_trajs.shape)
    print(lst_adj_matrix.shape)
    print(lst_dist.shape)
    # return
    return lst_ego_hists, lst_future_trajs, lst_neighbor_hists, lst_adj_matrix, lst_dist

def generate_training_data_gatsbi_one_batch(trajectory_data, batch_triplets, history_length=100, prediction_length=25, n_neighbors=5, max_permutations=32, data_type="train"):
    ego_hists = []
    predicted_trajs = []
    neighbor_hists_list = []
    adj_matrix_list = []
    dist_list = []
    for sequence, ego_vehicle_id, frame_id in batch_triplets:
        if not data_type=="train":
            max_permutations = 1
        for n_repetitions in range(0, max_permutations):
            speed_history_consideration = cs.SPEED_ESTIMATION_HORIZON
            neighbor_trajs, lane_xy, pred_traj, adj_matrix, dist = generate_training_data_gatsbi_one_record(trajectory_data, n_neighbors, history_length, prediction_length, sequence, ego_vehicle_id, frame_id, speed_history_consideration, data_type)
            if neighbor_trajs is None:
                continue
            ego_hists.append(lane_xy)
            predicted_trajs.append(pred_traj)
            neighbor_hists_list.append(np.stack(neighbor_trajs, axis=0))
            adj_matrix_list.append(adj_matrix)
            dist_list.append(dist)
    # conversion to tensors
    if len(ego_hists)==0:
        return None, None, None, None
    ego_hist = torch.tensor(np.stack(ego_hists, axis=0), dtype=torch.float32) 
    predicted_traj = torch.tensor(np.stack(predicted_trajs, axis=0), dtype=torch.float32)  
    neighbor_hists = torch.tensor(np.stack(neighbor_hists_list, axis=0), dtype=torch.float32) 
    adj_matrixs = torch.tensor(np.stack(adj_matrix_list, axis=0), dtype=torch.float32) 
    dist_list = torch.tensor(np.stack(dist_list, axis=0), dtype=torch.float32) 
    # Now you have:
    # ego_hist: (batch_size, history_length, 2)
    # predicted_traj: (batch_size, prediction_length, 2)
    # neighbor_hists: (batch_size, n_neighbours, history_length, 2)
    # adj_matrixs: (batch_size, n_neighbours+1, n_neighbours+1, 3)
    # dist: (batch_size, 1, 1)
    return ego_hist, predicted_traj, neighbor_hists, adj_matrixs, dist_list

def generate_training_data_gatsbi_one_record(trajectory_data, n_neighbors, history_length, prediction_length, sequence, ego_vehicle_id, frame_id, speed_history_consideration, data_type="train"):
    # find neighbors for this record
    relevant_neighbors = get_relevant_neighbors(trajectory_data, sequence, frame_id, ego_vehicle_id, n_neighbors)
    if data_type=="train":
        random.shuffle(relevant_neighbors)
    df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
    if df_trajectory is None:
        return None, None, None, None
    df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, history_length)
    df_veh_future = get_trajectory_future(df_trajectory, ego_vehicle_id, frame_id, prediction_length)
    # ego history
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
    # helper function
    def get_trajectory(neighbor_trajs, lane_xy, neigh_index):
        if neigh_index == n_neighbors:
            return lane_xy
        else:
            return neighbor_trajs[neigh_index]
    # adjacency matrix with four different features
    adj_matrix = []
    for n1 in range(0, n_neighbors+1):
        mat_n1 = []
        for n2 in range(0, n_neighbors+1):
            traj_1 = get_trajectory(neighbor_trajs, lane_xy, n1)
            traj_2 = get_trajectory(neighbor_trajs, lane_xy, n2)
            # distance
            curr_pos_1 = traj_1[-1]
            curr_pos_2 = traj_2[-1]
            distance = np.linalg.norm(curr_pos_1 - curr_pos_2)
            # angle
            delta = curr_pos_2 - curr_pos_1
            angle = np.arctan2(delta[1], delta[0])  # angle in radians
            # relative speed x [m/s]
            # relative speed y [m/s]
            dist_x_now = traj_1[-1][0] - traj_2[-1][0]
            dist_y_now = traj_1[-1][1] - traj_2[-1][1]
            dist_x_pre = traj_1[-1-speed_history_consideration][0] - traj_2[-1-speed_history_consideration][0]
            dist_y_pre = traj_1[-1-speed_history_consideration][1] - traj_2[-1-speed_history_consideration][1]
            rel_v_x = dist_x_now - dist_x_pre
            rel_v_y = dist_y_now - dist_y_pre
            # final feature vector        
            mat_n2 = [distance, angle, rel_v_x, rel_v_y]
            mat_n1.append(mat_n2)
        adj_matrix.append(mat_n1)
    adj_matrix = np.asarray(adj_matrix)
    # distance from road border
    distance = cs.CIRCLE_OUTER_RADIUS - lane_xy[-1][0]
    # return
    return neighbor_trajs, lane_xy, pred_traj, adj_matrix, distance