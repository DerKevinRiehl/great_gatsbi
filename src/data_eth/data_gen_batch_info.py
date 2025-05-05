"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
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

from data_eth.trajectory_loader_eth import transform_ego_perspective



# #############################################################################
# METHODS
def generate_data_batches(trajectory_data, batches):
    lst_batch_info = []
    for ego_vehicle_id, frame_id in tqdm(batches, desc="Loading Data"):
        # --- Find neighbors and transform to ego perspective
        df_trajectory = transform_ego_perspective(trajectory_data, ego_vehicle_id, frame_id)
        if df_trajectory is None:
            continue
        # --- Collect all
        lst_batch_info.append(["ETH", ego_vehicle_id, frame_id])
    # --- Stack to tensors
    data_dict = {
        'batch_info': lst_batch_info
    }
    return data_dict