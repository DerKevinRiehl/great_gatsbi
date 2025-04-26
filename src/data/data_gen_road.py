"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for road features.
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.trajectory_loader import transform_ego_perspective, get_trajectory_history
import utils.constants as cs




# #############################################################################
# METHODS
def generate_data_road(trajectory_data, batches):
    lst_dist = []

    for sequence, ego_vehicle_id, frame_id in tqdm(batches, desc="Loading Data"):
        # --- Transform to ego perspective
        df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
        if df_trajectory is None:
            continue

        # --- Ego history
        df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, cs.HISTORY_LENGTH)
        lane_xy = df_veh_history[["Lane_X", "Lane_Y"]].to_numpy()
        lane_xy = np.pad(lane_xy, ((max(0, cs.HISTORY_LENGTH - lane_xy.shape[0]), 0), (0, 0)), mode='constant')[-cs.HISTORY_LENGTH:]

        # --- Distance from road border
        distances = cs.CIRCLE_OUTER_RADIUS - lane_xy[-1:, 0]

        # --- Collect all
        lst_dist.append(torch.tensor(distances, dtype=torch.float32))

    # --- Stack to tensors
    data_dict = {
        'ego_road_border_distance': torch.stack(lst_dist)
    }
    return data_dict