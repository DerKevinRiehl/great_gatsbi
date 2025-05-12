"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to generate data for the batch infos for inference reconstruction.
"""




# #############################################################################
# IMPORTS
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from data.trajectory_loader import transform_ego_perspective




# #############################################################################
# METHODS
def generate_data_batches(trajectory_data, batches):
    lst_batch_info = []
    for sequence, ego_vehicle_id, frame_id in tqdm(batches, desc="Loading Data"):
        # --- Find neighbors and transform to ego perspective
        df_trajectory = transform_ego_perspective(trajectory_data, sequence, ego_vehicle_id, frame_id)
        if df_trajectory is None:
            continue
        # --- Collect all
        lst_batch_info.append([sequence, ego_vehicle_id, frame_id])
    # --- Stack to tensors
    data_dict = {
        'batch_info': lst_batch_info
    }
    return data_dict