"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains constants relevant for the codes / pipelines.
"""



# ### PREDICTION & DATA RELATED
PREDICTION_LENGTH = 100
HISTORY_LENGTH = 100 
BATCH_SIZE = 32
N_NEIGHBORS = 5

# ### TRAINING RELATED
N_EPOCHS = 10

# ### GEOMETRY RELATED
CIRCLE_OUTER_RADIUS = 15.0             # [m]
CIRCLE_TRUE_RADIUS = 5.0               # [m]
CIRCLE_FILTER_RADIUS_TOLL = 1.05       # [factor]

# ### VIDEO RELATED
FRAME_SKIP = 25*2                      # the trajectories are most plausible when cutting the tails by 2 seconds (50 frames)
FPS = 25                               # frame rate per second, recording of videos

# ### SEQUENCES
TRAINING_VIDEOS = {
    "DJI_20240906103036_0003_D.MP4",
    "DJI_20240906103442_0004_D.MP4",
    "DJI_20240906103850_0005_D.MP4",
    "DJI_20240906105321_0009_D.MP4",
    "DJI_20240906105621_0010_D.MP4",
    "DJI_20240906110432_0012_D.MP4"
}

TESTING_VIDEOS = {
    "DJI_20240906110027_0011_D.MP4",
}

VIDEOS_PARTS = {
    "DJI_20240906103036_0003_D.MP4": {
        "PART_1": [12*FPS, 78*FPS],    # 6 bikes
        "PART_2": [97*FPS, 138*FPS],   # 10 bikes
        "PART_3": [208*FPS, 214*FPS],  # 10 bikes
        "PART_4": [225*FPS, 6154],     # 14 bikes
    },
    "DJI_20240906103442_0004_D.MP4": {
        "PART_1": [0*FPS, 55*FPS],     # 14 bikes
        "PART_2": [114*FPS, 180*FPS],  # 19 bikes
    },    
    "DJI_20240906103850_0005_D.MP4": {
        "PART_1": [13*FPS, 82*FPS],    # 22 bikes
    },
    # "DJI_20240906104511_0007_D.MP4": {
    #     "PART_1": [18*FPS, 85*FPS],  # 20 bikes
    #     "PART_2": [137*FPS, 210*FPS],# 20 bikes
    #     "PART_3": [210*FPS, 6128],   # 20 bikes
    # },
    # "DJI_20240906104917_0008_D.MP4": {
    #     "PART_1": [71*FPS, 131*FPS], # 19 bikes
    #     "PART_2": [172*FPS, 208*FPS],# 20 bikes
    # },
    "DJI_20240906105321_0009_D.MP4": {
        "PART_1": [6*FPS, 14*25],      # 13 bikes
    },
    "DJI_20240906105621_0010_D.MP4": {
        "PART_1": [14*FPS, 37*FPS],    # 6 bikes
        "PART_2": [50*FPS, 76*FPS],    # 9 bikes
        "PART_3": [90*FPS, 115*FPS],   # 12 bikes
        "PART_4": [123*FPS, 130*FPS],  # 16 bikes
        "PART_5": [130*FPS, 148*FPS],  # 17 bikes
        "PART_6": [238*FPS, 6138]      # 17 bikes
    },
    "DJI_20240906110027_0011_D.MP4": {
        "PART_1": [0*FPS, 69*FPS],     # 17 bikes
        "PART_2": [101*FPS, 132*FPS],  # 17 bikes
        "PART_3": [140*FPS, 175*FPS],  # 17 bikes
        "PART_4": [187*FPS, 220*FPS],  # 17 bikes
        "PART_5": [234*FPS, 6122],     # 17 bikes
    },
    "DJI_20240906110432_0012_D.MP4": {
        "PART_1": [0*FPS, 25*FPS],     # 17 bikes
    },
}

# ### GATSBI RELATED
SPEED_ESTIMATION_HORIZON = 25
