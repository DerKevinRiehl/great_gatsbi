"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains availability of models with different multimodality.
"""




# #############################################################################
# MODEL AVAILABILITIES

ML_MODELS_UNIMODAL = [
    "ego_lstm",   
    "social_lstm",
    "social_bigat",
    "gatsbi_physics_module",
    "gatsbi_social_module",
    "gatsbi",
    "gatsbi_abl_anticip",
    "gatsbi_abl_star",
    "gatsbi_abl_decay",
    "gatsbi_abl_phy_anticip",
    "gatsbi_abl_phy_star",
    "gatsbi_abl_phy_decay",
    "gatsbi_abl_phy",
]

ML_MODELS_MULTIMODAL_GMM = [    
    "ego_lstm",   
    "social_lstm",
    "social_bigat",
    "gatsbi_physics_module",
    "gatsbi_social_module",
    "gatsbi",
    "gatsbi_abl_anticip",
    "gatsbi_abl_star",
    "gatsbi_abl_decay",
    "gatsbi_abl_phy_anticip",
    "gatsbi_abl_phy_star",
    "gatsbi_abl_phy_decay",
    "gatsbi_abl_phy",
]

ML_MODELS_MULTIMODAL_CVAE = [
]