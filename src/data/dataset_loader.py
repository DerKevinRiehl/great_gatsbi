"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
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
    data_batches = torch.load('../data/2_datasets/data_batches_'+sequence+'.pt')
    data = {}    
    for data_set in [data_social, data_physics, data_road]:
        for key in data_set:
            data[key] = data_set[key]
    return data, data_batches

def load_data_from_videos(lst_videos):
    dataset = None
    batches_tot = {}
    for relevant_video in lst_videos:
        for relevant_part in cs.VIDEOS_PARTS[relevant_video]:
            data_sequence, sequence_batches = load_data_for_sequence(sequence=relevant_video+"-"+relevant_part)
            batches_tot[relevant_video+"-"+relevant_part] = sequence_batches
            if dataset is None:
                dataset = data_sequence.copy()
            else:
                for key in dataset:
                    dataset[key] = torch.cat((dataset[key], data_sequence[key]), dim=0)
    return dataset, batches_tot

def load_data_from_videos_inference(sequence):
    dataset = None
    batches_tot = {}
    relevant_video = sequence.split("-")[0]
    relevant_part = sequence.split("-")[1]
    data_sequence, sequence_batches = load_data_for_sequence(sequence=relevant_video+"-"+relevant_part)
    batches_tot[relevant_video+"-"+relevant_part] = sequence_batches
    if dataset is None:
        dataset = data_sequence.copy()
    else:
        for key in dataset:
            dataset[key] = torch.cat((dataset[key], data_sequence[key]), dim=0)
    return dataset, batches_tot

def load_dataset(model_name, videos, prediction_length):
    dataset, batches_tot = load_data_from_videos(videos)
    return unpack_data(dataset, model_name, prediction_length)

def load_dataset_inference(model_name, sequence, prediction_length):
    dataset, batches_tot = load_data_from_videos_inference(sequence)
    return batches_tot[sequence]["batch_info"], unpack_data(dataset, model_name, prediction_length)



