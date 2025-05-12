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




# #############################################################################
# METHODS
def load_data_for_source(source):
    data_social = torch.load('../data/2_datasets/data_social_'+source+'.pt')
    data_physics = torch.load('../data/2_datasets/data_physics_'+source+'.pt')
    data_batches = torch.load('../data/2_datasets/data_batches_'+source+'.pt')
    data = {}    
    for data_set in [data_social, data_physics]:
        for key in data_set:
            data[key] = data_set[key]
    return data, data_batches

def load_data_from_source(source):
    dataset = None
    batches_tot = {}
    data_sequence, sequence_batches = load_data_for_source(source)
    batches_tot[source] = sequence_batches
    if dataset is None:
        dataset = data_sequence.copy()
    else:
        for key in dataset:
            dataset[key] = torch.cat((dataset[key], data_sequence[key]), dim=0)
    return dataset, batches_tot

def load_data_from_source_inference(source):
    dataset = None
    batches_tot = {}
    data_sequence, sequence_batches = load_data_for_source(source)
    batches_tot[source] = sequence_batches
    if dataset is None:
        dataset = data_sequence.copy()
    else:
        for key in dataset:
            dataset[key] = torch.cat((dataset[key], data_sequence[key]), dim=0)
    return dataset, batches_tot

def load_dataset(model_name, source, prediction_length):
    dataset, batches_tot = load_data_from_source(source)
    return unpack_data(dataset, model_name, prediction_length)

def load_dataset_inference(model_name, source, prediction_length):
    dataset, batches_tot = load_data_from_source_inference(source)
    return batches_tot[source]["batch_info"], unpack_data(dataset, model_name, prediction_length)



