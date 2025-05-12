"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for physics features.
"""




# #############################################################################
# IMPORTS
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.model_classic import ModelClassic, constant_velocity_predictor, constant_acceleration_predictor
from models.model_bike_kinematics import ModelBikeKinematics
from models.model_ekf import ModelXKalman
import utils.constants as cs

from data_eth.trajectory_loader_eth import transform_ego_perspective, get_trajectory_history, get_trajectory_future



# #############################################################################
# METHODS
def generate_data_physics(trajectory_data, batches):
    lst_ego_hists = []
    lst_future_trajs = []
    lst_pred_cv = []
    lst_pred_ca = []
    lst_pred_bk = []
    lst_pred_xk = []

    for ego_vehicle_id, frame_id in tqdm(batches, desc="Loading Data"):
        # --- Data extraction ---
        df_trajectory = transform_ego_perspective(
            trajectory_data, ego_vehicle_id, frame_id
        )
        if df_trajectory is None:
            continue
        df_veh_history = get_trajectory_history(
            df_trajectory, ego_vehicle_id, frame_id, cs.HISTORY_LENGTH_HOTEL
        )
        df_veh_future = get_trajectory_future(
            df_trajectory, ego_vehicle_id, frame_id, cs.PREDICTION_LENGTH_HOTEL
        )

        # --- Ego history ---
        ego_hist = df_veh_history[["Lane_X", "Lane_Y"]].to_numpy()
        if ego_hist.shape[0] < cs.HISTORY_LENGTH_HOTEL:
            pad_len = cs.HISTORY_LENGTH_HOTEL - ego_hist.shape[0]
            ego_hist = np.pad(ego_hist, ((pad_len, 0), (0, 0)), mode='constant')
        else:
            ego_hist = ego_hist[-cs.HISTORY_LENGTH_HOTEL:]

        # --- Future trajectory ---
        pred_traj = df_veh_future[["Lane_X", "Lane_Y"]].to_numpy()
        if pred_traj.shape[0] < cs.PREDICTION_LENGTH_HOTEL:
            pad_len = cs.PREDICTION_LENGTH_HOTEL - pred_traj.shape[0]
            pred_traj = np.pad(pred_traj, ((0, pad_len), (0, 0)), mode='constant')
        else:
            pred_traj = pred_traj[:cs.PREDICTION_LENGTH_HOTEL]

        # --- Model predictions ---
        pred_cv = ModelClassic(model_func=constant_velocity_predictor, prediction_length=cs.PREDICTION_LENGTH_HOTEL)([ego_hist])[0]
        pred_ca = ModelClassic(model_func=constant_acceleration_predictor, prediction_length=cs.PREDICTION_LENGTH_HOTEL)([ego_hist])[0]
        pred_bk = ModelBikeKinematics(prediction_length=cs.PREDICTION_LENGTH_HOTEL)([ego_hist])[0]
        pred_xk = ModelXKalman(prediction_length=cs.PREDICTION_LENGTH_HOTEL)(np.expand_dims(ego_hist, axis=0))[0]

        # --- Collect results ---
        lst_ego_hists.append(torch.tensor(ego_hist, dtype=torch.float32))
        lst_future_trajs.append(torch.tensor(pred_traj, dtype=torch.float32))
        lst_pred_cv.append(torch.tensor(pred_cv, dtype=torch.float32))
        lst_pred_ca.append(torch.tensor(pred_ca, dtype=torch.float32))
        lst_pred_bk.append(torch.tensor(pred_bk, dtype=torch.float32))
        lst_pred_xk.append(torch.tensor(pred_xk, dtype=torch.float32))

    # --- Stack tensors ---
    if not lst_ego_hists:
        return None  # or return dict with None values

    data_dict = {
        'preds_cv': torch.stack(lst_pred_cv),
        'preds_ca': torch.stack(lst_pred_ca),
        'preds_bk': torch.stack(lst_pred_bk),
        'preds_xk': torch.stack(lst_pred_xk),
    }
    return data_dict