"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for social features (interactions).
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import utils.constants as cs

from data_eth.trajectory_loader_eth import get_relevant_neighbors, transform_ego_perspective, get_trajectory_history, get_trajectory_future


# #############################################################################
# METHODS
def generate_data_social(trajectory_data, batches):
    lst_ego_hists = []
    lst_future_trajs = []
    lst_neighbor_hists = []
    lst_adj_matrix = []

    for ego_vehicle_id, frame_id in tqdm(batches, desc="Loading Data"):
        # --- Find neighbors and transform to ego perspective
        relevant_neighbors = get_relevant_neighbors(trajectory_data, frame_id, ego_vehicle_id, cs.N_NEIGHBORS)
        df_trajectory = transform_ego_perspective(trajectory_data, ego_vehicle_id, frame_id)
        if df_trajectory is None:
            continue

        # --- Ego history
        df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, cs.HISTORY_LENGTH_HOTEL)
        lane_xy = df_veh_history[["Lane_X", "Lane_Y"]].to_numpy()
        lane_xy = np.pad(lane_xy, ((max(0, cs.HISTORY_LENGTH_HOTEL - lane_xy.shape[0]), 0), (0, 0)), mode='constant')[-cs.HISTORY_LENGTH_HOTEL:]

        # --- Future trajectory
        df_veh_future = get_trajectory_future(df_trajectory, ego_vehicle_id, frame_id, cs.PREDICTION_LENGTH_HOTEL)
        pred_traj = df_veh_future[["Lane_X", "Lane_Y"]].to_numpy()
        pred_traj = np.pad(pred_traj, ((0, max(0, cs.PREDICTION_LENGTH_HOTEL - pred_traj.shape[0])), (0, 0)), mode='constant')[:cs.PREDICTION_LENGTH_HOTEL]

        # --- Neighbor histories
        neighbor_trajs = []
        for neighbor_id in relevant_neighbors:
            df = get_trajectory_history(df_trajectory, neighbor_id, frame_id, cs.HISTORY_LENGTH_HOTEL)
            arr = df[["Lane_X", "Lane_Y"]].to_numpy()
            arr = np.pad(arr, ((max(0, cs.HISTORY_LENGTH_HOTEL - arr.shape[0]), 0), (0, 0)), mode='constant')[-cs.HISTORY_LENGTH_HOTEL:]
            neighbor_trajs.append(arr)
        while len(neighbor_trajs) < cs.N_NEIGHBORS:
            neighbor_trajs.append(np.zeros((cs.HISTORY_LENGTH_HOTEL, 2), dtype=np.float32))
        neighbor_trajs = neighbor_trajs[:cs.N_NEIGHBORS]

        # --- Adjacency matrix
        speed_history_consideration = cs.SPEED_ESTIMATION_HORIZON
        def get_traj(idx):
            return lane_xy if idx == cs.N_NEIGHBORS else neighbor_trajs[idx]
        adj_matrix = []
        for n1 in range(cs.N_NEIGHBORS + 1):
            row = []
            for n2 in range(cs.N_NEIGHBORS + 1):
                traj_1 = get_traj(n1)
                traj_2 = get_traj(n2)
                curr_pos_1 = traj_1[-1]
                curr_pos_2 = traj_2[-1]
                distance = np.linalg.norm(curr_pos_1 - curr_pos_2)
                delta = curr_pos_2 - curr_pos_1
                angle = np.arctan2(delta[1], delta[0])
                dist_x_now = traj_1[-1][0] - traj_2[-1][0]
                dist_y_now = traj_1[-1][1] - traj_2[-1][1]
                try:
                    dist_x_pre = traj_1[-1-speed_history_consideration][0] - traj_2[-1-speed_history_consideration][0]
                    dist_y_pre = traj_1[-1-speed_history_consideration][1] - traj_2[-1-speed_history_consideration][1]
                    rel_v_x = dist_x_now - dist_x_pre
                    rel_v_y = dist_y_now - dist_y_pre
                except:
                    rel_v_x = -1
                    rel_v_y = -1
                row.append([distance, angle, rel_v_x, rel_v_y])
            adj_matrix.append(row)
        adj_matrix = np.asarray(adj_matrix)

        # --- Collect all
        lst_ego_hists.append(torch.tensor(lane_xy, dtype=torch.float32))
        lst_future_trajs.append(torch.tensor(pred_traj, dtype=torch.float32))
        lst_neighbor_hists.append(torch.tensor(np.stack(neighbor_trajs, axis=0), dtype=torch.float32))
        lst_adj_matrix.append(torch.tensor(adj_matrix, dtype=torch.float32))

    # --- Stack to tensors
    data_dict = {
        'ego_trajectory_history': torch.stack(lst_ego_hists),
        'ego_trajectory_future': torch.stack(lst_future_trajs),
        'neighbor_trajectory_history': torch.stack(lst_neighbor_hists),
        'neighbor_adjacency_matrix': torch.stack(lst_adj_matrix),
    }
    return data_dict