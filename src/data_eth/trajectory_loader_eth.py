"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to load trajectories from the trajectory files of ETH pedestrian dataset,
calculate the lane coordinates, do coordinate transforms, and determine neighbors.
"""




# #############################################################################
# IMPORTS
# #############################################################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import utils.constants as cs



# #############################################################################
# METHODS
# #############################################################################

def load_ETH():
    tbl = pd.read_csv(
        "../data/X_other_datasets/seq_pedest_eth/obsmat.txt",
        sep=r'\s+',
        engine='python',
        header=None,
        names=["Frame_ID", "Vehicle_ID", "Lane_X", "Lane_Z", "Lane_Y", "v_x", "v_z", "v_y"]
    )
    tbl["Frame_ID"] = tbl["Frame_ID"].astype(int)
    tbl["Global_Time"] = tbl["Frame_ID"]*(1/2.5)
    tbl["v_Vel"] = np.hypot(tbl["v_x"], tbl["v_y"])
    tbl = tbl[["Vehicle_ID", "Frame_ID", "Global_Time", "v_Vel", "Lane_X", "Lane_Y"]]
    return tbl

def load_HOTEL():
    tbl = pd.read_csv(
        "../data/X_other_datasets/seq_pedest_hotel/obsmat.txt",
        sep=r'\s+',
        engine='python',
        header=None,
        names=["Frame_ID", "Vehicle_ID", "Lane_X", "Lane_Z", "Lane_Y", "v_x", "v_z", "v_y"]
    )
    tbl["Frame_ID"] = tbl["Frame_ID"].astype(int)
    tbl["Frame_ID"] = tbl["Frame_ID"]
    tbl["Global_Time"] = tbl["Frame_ID"]*(1/2.5)
    tbl["v_Vel"] = np.hypot(tbl["v_x"], tbl["v_y"])
    tbl = tbl[["Vehicle_ID", "Frame_ID", "Global_Time", "v_Vel", "Lane_X", "Lane_Y"]]
    return tbl


def get_unique_vehicles(trajectory_data):
    df_trajectory = trajectory_data.copy()
    unique_vehicles = df_trajectory["Vehicle_ID"].unique().tolist()
    return unique_vehicles

def get_frame_range(trajectory_data):
    df_trajectory = trajectory_data.copy()
    unique_frames = df_trajectory["Frame_ID"].unique().tolist()    
    return unique_frames[0] + cs.FRAME_SKIP, unique_frames[-1] - cs.FRAME_SKIP  

def transform_ego_perspective(trajectory_data, ego_vehicle_id, frame_id):
    df_trajectory = trajectory_data.copy()
    df_veh = df_trajectory[df_trajectory["Vehicle_ID"]==ego_vehicle_id]
    df_veh = df_veh[df_veh["Frame_ID"]==frame_id]
    if len(df_veh)==0:
        return None
    ref_x = df_veh["Lane_X"].iloc[0]
    ref_y = df_veh["Lane_Y"].iloc[0]
    df_trajectory["Lane_X"] = df_trajectory["Lane_X"] - ref_x
    df_trajectory["Lane_Y"] = df_trajectory["Lane_Y"] - ref_y
    return df_trajectory

def get_trajectory_history(df_trajectory, vehicle_id, frame_id, history_length):
    df_veh_history = df_trajectory[df_trajectory["Vehicle_ID"]==vehicle_id]
    df_veh_history = df_veh_history[df_veh_history["Frame_ID"]<=frame_id]
    df_veh_history = df_veh_history[df_veh_history["Frame_ID"]>frame_id-history_length]
    return df_veh_history

def get_trajectory_future(df_trajectory, vehicle_id, frame_id, future_length):
    df_veh_future = df_trajectory[df_trajectory["Vehicle_ID"]==vehicle_id]
    df_veh_future = df_veh_future[df_veh_future["Frame_ID"]<=frame_id+future_length]
    df_veh_future = df_veh_future[df_veh_future["Frame_ID"]>frame_id]
    return df_veh_future

def get_trajectory_from_df(df_trajectory):
    return df_trajectory["Lane_X"].tolist(), df_trajectory["Lane_Y"].tolist(), df_trajectory["v_Vel"].tolist()

def get_relevant_neighbors(trajectory_data, frame_id, ego_vehicle_id, n_neighbors = 5, max_dist = 20):
    df_trajectory = transform_ego_perspective(trajectory_data, ego_vehicle_id, frame_id)
    if df_trajectory is None:
        return []
    unique_vehicles = df_trajectory["Vehicle_ID"].unique().tolist()
    vehicle_pos = []
    for vehicle_id in unique_vehicles:
        df_veh = df_trajectory[df_trajectory["Vehicle_ID"]==vehicle_id]
        df_veh = df_veh[df_veh["Frame_ID"]==frame_id]
        if len(df_veh)>0:
            veh_pos = np.asarray([df_veh.iloc[0]["Lane_X"], df_veh.iloc[0]["Lane_Y"]])
        else:
            continue
        vehicle_pos.append([vehicle_id, np.linalg.norm(veh_pos)])
    filtered1 = [b for b in vehicle_pos if b[0] != ego_vehicle_id]
    filtered2 = [b for b in filtered1 if b[1] <= max_dist]
    closest_neighbors = sorted(filtered2, key=lambda x: x[1])[:n_neighbors]
    return [v[0] for v in closest_neighbors]
