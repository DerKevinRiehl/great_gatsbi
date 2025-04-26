"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains methods to load data from datasets files.
"""




# #############################################################################
# ### IMPORTS
import torch
import warnings
warnings.filterwarnings("ignore")

from models.model_loader import unpack_data
import utils.constants as cs




# #############################################################################
# METHODS
def load_data_for_sequence(sequence):
    data_social = torch.load('../data/2_datasets/data_social_'+sequence+'.pt')
    data_physics = torch.load('../data/2_datasets/data_physics_'+sequence+'.pt')
    data_road = torch.load('../data/2_datasets/data_road_'+sequence+'.pt')
    data = {}    
    for data_set in [data_social, data_physics, data_road]:
        for key in data_set:
            data[key] = data_set[key]
    return data

def load_data_from_videos(lst_videos):
    training_data = None
    for relevant_video in lst_videos:
        for relevant_part in cs.VIDEOS_PARTS[relevant_video]:
            training_data_sequence = load_data_for_sequence(sequence=relevant_video+"-"+relevant_part)
            if training_data is None:
                training_data = training_data_sequence.copy()
            else:
                for key in training_data:
                    training_data[key] = torch.cat((training_data[key], training_data_sequence[key]), dim=0)
    return training_data

def load_dataset(model_name, videos, prediction_length):
    dataset = load_data_from_videos(videos)
    return unpack_data(dataset, model_name, prediction_length)

