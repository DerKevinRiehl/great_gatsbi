"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to load trajectories from the trajectory files,
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
def load_trajectories(relevant_cols = True):
    trajectory_data = {}
    for relevant_video in cs.VIDEOS:
        for relevant_part in cs.VIDEOS[relevant_video]:
            sequence = relevant_video+"-"+relevant_part
            df_trajectory = pd.read_csv("../data/1_trajectories/"+relevant_video+"_"+relevant_part+".txt")
            # Unwrap Angle
            unique_vehicles = df_trajectory["Vehicle_ID"].unique().tolist()
            data = []
            for vehicle_id in unique_vehicles:
                df_veh_id = df_trajectory[df_trajectory["Vehicle_ID"]==vehicle_id]
                df_veh_id["Polar_X"] = np.unwrap(df_veh_id["Polar_X"].tolist())
                data.append(df_veh_id)
            df_trajectory_unwrapped = pd.concat(data, ignore_index=True)
            df_trajectory_unwrapped = df_trajectory_unwrapped.sort_values(by=["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
            df_trajectory = df_trajectory_unwrapped
            # Calculate Lane Coordinates
            df_trajectory["Lane_X"] = df_trajectory["Polar_Y"]
            df_trajectory["Lane_Y"] = df_trajectory["Polar_X"]*cs.CIRCLE_OUTER_RADIUS
            if relevant_cols:
                df_trajectory = df_trajectory[["Vehicle_ID", "Frame_ID", "Global_Time", "v_Vel", "Lane_X", "Lane_Y"]]
            trajectory_data[sequence] = df_trajectory
    return trajectory_data

def get_unique_vehicles(trajectory_data, sequence):
    df_trajectory = trajectory_data[sequence].copy()
    unique_vehicles = df_trajectory["Vehicle_ID"].unique().tolist()
    return unique_vehicles

def get_frame_range(trajectory_data, sequence):
    df_trajectory = trajectory_data[sequence].copy()
    unique_frames = df_trajectory["Frame_ID"].unique().tolist()    
    return unique_frames[0] + cs.FRAME_SKIP, unique_frames[-1] - cs.FRAME_SKIP  

def transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id):
    df_trajectory = trajectory_data[sequence].copy()
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

def get_relevant_neighbors(trajectory_data, sequence, frame_id, ego_vehicle_id, n_neighbors = 5, max_dist = 20):
    df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
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



"""
ego_vehicle_id = "BICYCLE_24" # "BICYCLE_24"
frame_id = 425 #441 # 1000
sequence = "DJI_20240906103850_0005_D.MP4-PART_1" # "DJI_20240906103036_0003_D.MP4-PART_1" # "DJI_20240906103850_0005_D.MP4-PART_1"
history_length = 100 # 25
prediction_length = 25

trajectory_data = load_trajectories()
df_trajectory = trajectory_data[sequence]
df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
df_veh_history = get_trajectory_history(df_trajectory, ego_vehicle_id, frame_id, history_length)
"""



