>📋  Based on template README.md for code accompanying a Machine Learning paper from [paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) 

# Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#models)
- [Results](#results)
- [License](#license)
- [Cluster and Runtime](#cluster)

## [Introduction](#introduction)




## [Repository Structure](#repository-structure)

```
./neurips25_great_gatsbi/
├── data/
│   ├── 0_videos/
│   ├── 1_trajectories/
│   ├── 2_training_datasets/
│   ├── 3_testing_datasets/
│   ├── 4_models/
│   └── 5_inferences/
├── figures/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── main.py
├── LICENSE
├── README.md
└── requirements.txt
```

## [Dataset](#dataset)

<table>
<tr>
<td>
Specifically for this project we conducted a <b>mass-cycling experiment</b> during a conference workshop at our university, and video-captured the experiment with a drone from above. 
We cooperated with a company that wanted to promote their rental bicycles, and they kindly provided us with <b>more than 25 bicycles</b>.
</td>
<td>
<img src="figures/mass_cycling_experiment.PNG" />
</td>
</tr>
</table>


<details>
We chose a specific location at our campus for the experiment, that offers a **circular track (ring road)**.
This has several advantages: 
(i) we can observe all bicycles at the same time using a drone, 
(ii) the bicycles have a homogeneous road, solely interactions between bicycles drive their behaviour (and following the right alignment principle), 
(iii) we could control the traffic density on the road by adding or removing bicycles and disrupting the traffic flow dynamics manually.

In total we recorded **9 video files**, that **cover 30 minutes** of recording, at a resolution of 3840x2160 pixels, and a framerate of 25 frames/second.
The videos were stored in MP4 format and are about 25.7 GB large.
Due to interruptions by trucks, cars, and drone landing for battery change, only some parts of the video are useful for the purpose of this investigation.
We therefore selected **20 sequences** from these videos, as outlined in the following table.

| sequence_nr | video_file                    | part    | from_frame | to_frame | num_frames | num_bicycles | data  |
|-------------|-------------------------------|---------|------------|----------|------------|--------------|-------|
| 1           | DJI_20240906103036_0003_D.MP4 | PART_1  | 300        | 1950     | 1650       | 6            | train |
| 2           | DJI_20240906103036_0003_D.MP4 | PART_2  | 2425       | 3450     | 1025       | 10           | train |
| 3           | DJI_20240906103036_0003_D.MP4 | PART_3  | 5200       | 5350     | 150        | 10           | train |
| 4           | DJI_20240906103036_0003_D.MP4 | PART_4  | 5625       | 6154     | 529        | 14           | train |
| 5           | DJI_20240906103442_0004_D.MP4 | PART_1  | 0          | 1375     | 1375       | 14           | train |
| 6           | DJI_20240906103442_0004_D.MP4 | PART_2  | 2850       | 4500     | 1650       | 19           | train |
| 7           | DJI_20240906103850_0005_D.MP4 | PART_1  | 325        | 2050     | 1725       | 22           | train |
| 8           | DJI_20240906105321_0009_D.MP4 | PART_1  | 150        | 350      | 200        | 13           | train |
| 9           | DJI_20240906105621_0010_D.MP4 | PART_1  | 350        | 925      | 575        | 6            | train |
| 10          | DJI_20240906105621_0010_D.MP4 | PART_2  | 1250       | 1900     | 650        | 9            | train |
| 11          | DJI_20240906105621_0010_D.MP4 | PART_3  | 2250       | 2875     | 625        | 12           | train |
| 12          | DJI_20240906105621_0010_D.MP4 | PART_4  | 3075       | 3250     | 175        | 16           | train |
| 13          | DJI_20240906105621_0010_D.MP4 | PART_5  | 3250       | 3700     | 450        | 17           | train |
| 14          | DJI_20240906105621_0010_D.MP4 | PART_6  | 5950       | 6138     | 188        | 17           | train |
| 15          | DJI_20240906110027_0011_D.MP4 | PART_1  | 0          | 1725     | 1725       | 17           | test  |
| 16          | DJI_20240906110027_0011_D.MP4 | PART_2  | 2525       | 3300     | 775        | 17           | test  |
| 17          | DJI_20240906110027_0011_D.MP4 | PART_3  | 3500       | 4375     | 875        | 17           | test  |
| 18          | DJI_20240906110027_0011_D.MP4 | PART_4  | 4675       | 5500     | 825        | 17           | test  |
| 19          | DJI_20240906110027_0011_D.MP4 | PART_5  | 5850       | 6122     | 272        | 17           | test  |
| 20          | DJI_20240906110432_0012_D.MP4 | PART_1  | 0          | 625      | 625        | 17           | train  |
| | | | | | | |
| | | | | **total** | 16054 | 25 | |

In the next step we used two different **computer vision approaches** and manual annotation to **detect bicycles on the aerial images**: (i) object detection with YOLO, (ii) an approach that compares two consecutive frames for differences to identify moving objects with OpenCV.
Also, we extracted a characteristic pattern (the inner circle) with known geometric properties (radius 5.0m) using **Hough transform** (OpenCV), in order to conduct a homography transformation from pixel to Cartesian coordinates.
Afterwards we used a computational pipeline to extract trajectories from these object detections.
The trajectories were filtered with a **Kalman-Filter** and checked for quality manually. 

The trajectories can be found in `\neurips25_great_gatsbi\data\1_trajectories`.
The trajectories of each sequence (named `video_file`+`-`+`part`+`.txt`) are stored in a **csv format**, with following columns:

| column_name   | example value             |  unit |
|---------------|---------------------------|-------|
| Vehicle_ID    | BICYCLE_1                 |   -   |
| Frame_ID      | 300                       |   -   |
| Global_Time   | 12.0                      |   [s] |
| Cartesian_X   | 1.969604762847424         |   [m] |
| Cartesian_Y   | 14.569321533923302        |   [m] |
| Polar_X       | 4.846762832391482         | [rad] |
| Polar_Y       | 14.701852702318593        |   [m] |
| v_Length      | 1.8                       |   [m] |
| v_Width       | 0.64                      |   [m] |
| v_Vel         | 0.7664797280991118        | [m/s] |

Please note following assumpetions when creating this dataset:
- the length and width were fixed for every bicycle.
- the Cartesian coodinates are relative to the center of the circle of our road track 
- Polar_X represents the angle and Polar_Y the radius (distance to circle center)

The volunteering participants of this mass cycling experiment all were informed that they will be recorded and gave their written consent.

</details>

## [Requirements](#requirements)


### Python & Packages
The implementation is conducted in **Python** (version >3.7).
To install requirements, please use the package management system **pip** as follows:

```setup
pip install -r requirements.txt
```

### Computational Resources
The proposed network was implemented in Pytorch which allows for the use of CPU on your local machine, in case you don't have access to any GPUs.
In case GPUs are available, the implementation will automatically switch to use CUDA.
Within a reasonable amount of time, training and testing can be conducted even without GPUs.

### Preparation of Trajectory Dataset
Please extract all trajectory txt files from `\neurips25_great_gatsbi\data\1_trajectories\1_trajectories.zip` and store them in the folder `\neurips25_great_gatsbi\data\1_trajectories`.

On Linux you could use this command:
```
unzip neurips25_great_gatsbi/data/1_trajectories/1_trajectories.zip -d neurips25_great_gatsbi/data/1_trajectories/
```

## [Training](#training)

### Data Generation
We recommend to **precalculate all training & testing data from the trajectory data**, as this is time consuming (especially physical and social features) this might take up to 20 hours.
We recommend reviewers to run it for one video only with few frames (e.g. *PART_3* of *DJI_20240906103036_0003_D.MP4*)
First, training data needs to be generated with the script `data_generator.py`. 
The results are stored in `\neurips25_great_gatsbi\data\2_datasets`.

The script can be used as follows:
```
python data_generator.py
```

Three different types of features are generated:
- **Physical Features**
    - preditions according to constant velocity model
    - predictions according to constant acceleration model
    - predictions according to bicycle kinematics model
    - predictions according to an extended Kalman filter
- **Social Features**
    - ego's historical trajectory
    - ego's future trajectory (for testing only)
    - neighbor's historical trajectory
    - adjacency matrix representing ego and neighbor's graph incl. distance, angle, rel. speed x and y
    - neighbors include ego's five closest neighbors (within a max. distance of 20m)
- **Road Features**
    - ego's historical distance from road edge

### Model Training

To train the model you can use  the script `train_model.py`. 
The resulting models are stored in `\neurips25_great_gatsbi\data\4_models`.

The script can be used as follows:
```
python train_model.py [1] [2] [3]
    [1] - model ("social_lstm" or "gatsbi" or "physics_lstm")
    [2] - prediction_length in [s] (25, 50 , 75, 100)
    [3] - max_epochs
```

An example to train a model can be found here:
```
python train_model.py social_lstm 25 50
```

After training of each epoch a model file is stored in the models folder, as well as a txt file containing the performance on the test set.


## [Evaluation](#evaluation)

To test the model (and thus evaluate) you can use the script `test_model.py`. 
The resulting evaluation metrics are printed to the console.

The script can be used as follows:
```eval
python test_model.py [1] [2] [3]
    [1] - model ("social_lstm" or "gatsbi" or "const_v" or "const_a" or "kinematics" or "xkalman" or "physics_lstm")
    [2] - model_file_name
    [3] - prediction_length in [s] (25, 50, 75, 100)
```

An example to run a test can be found here:
```
python test_model_all.py social_lstm social_lstm_25_5_0010.model 25
```
This outputs something like this:
```
{'ADE': 0.19542816281318665, 'FDE': 0.4298399090766907}
```

### Evaluation Metrics

We use average displacement error (ADE) and final displacement error (FDE) as evaluation metrics.
These evaluation metrics are common metrics in the domain of trajectory prediction.



## [Pre-trained Models](#models)

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## [Results](#results)


### Benchmark results
The benchmark of different models shows that the proposed GATsBi model is outperforming pedestrian specific (social_lstm, social_bigats) and car specific (const_v, const_a) models.

| Model  | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |
|------------|----|----|----|----|----|----|----|----|
| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |
| **car specific (physics)** |   |   |   |   |   |   |   |   |
| const_v | 0.1080 | 0.2818 | 0.5460 | 0.9406 | 0.2592 | 0.6568 | 1.5245 | 2.7275 |
|       |[0.0076]|[0.0194]|[0.0444]|[0.1059]|[0.0182]|[0.0436]|[0.1787]|[0.4278]|
| const_a | 0.1281 | 0.5504 | 1.2951 | 2.3929 | 0.3934 | 1.6373 | 4.0117 | 7.3837 |
|       |[0.0118]|[0.0482]|[0.1180]|[0.2292]|[0.0346]|[0.1422]|[0.3857]|[0.7451]|
| kinematics | 0.1103 | 0.3942 | 0.8914 | 1.6309 | 0.3027 | 1.1047 | 2.7238 | 4.9800 |
|       |[0.0088]|[0.0364]|[0.0905]|[0.1795]|[0.0260]|[0.1068]|[0.3056]|[0.5935]|
| xkalman | 0.1445 | 0.3269 | 0.5967 | 0.9948 | 0.3068 | 0.7154 | 1.5887 | 2.7913 |
|       |[0.0122]|[0.0242]|[0.0512]|[0.1146]|[0.0235]|[0.0492]|[0.1904]|[0.4417]|
| **pedestrian specific** |   |   |   |   |   |   |   |   |
| social_lstm | 0.0871 | 0.2413 | 0.4755 | 0.8203 | 0.2135 | 0.5390 | 1.3003 | 2.3724 |
|       | [0.0057] | [0.0175] | [0.0372] | [0.0837] | [0.0152] | [0.0353] | [0.1648] | [0.3723] |
| social_bigats |   |   |   |   |   |   |   |   |
|      . |   |   |   |   |   |   |   |   |
| **own models** |   |   |   |   |   |   |   |   |
| physics_lstm | 0.0744 | 0.2157 | 0.4324 | 0.7724 | 0.1967 | 0.5082 | 1.2479 | 2.3160 |
|       | [0.0068] | [0.0148] | [0.0377] | [0.0987] | [0.0167] | [0.0322] | [0.1695] | [0.4164] |
| gatsbiv1 | 0.0763 | 0.2181 | 0.4222 | 0.7812 | 0.1955 | 0.4996 | 1.2218 | 2.3137 |
|       | [0.0031] | [0.0167] | [0.0342] | [0.1045] | [0.0091] | [0.0264] | [0.1548] | [0.4081] |
| gatsbiv2 (best)| 0.0795 | 0.2228 | 0.4261 | 0.7366 | 0.2033 | 0.5107 | 1.1500 | 2.1162 |



>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

### Codes to reproduce

#### 1) Create Log Files Of Classical Models
```
python test_model.py const_v x 25 all > ../data/3_logs/const_v_25.txt
python test_model.py const_a x 25 all > ../data/3_logs/const_a_25.txt
python test_model.py kinematics x 25 all > ../data/3_logs/kinematics_25.txt
python test_model.py xkalman x 25 all > ../data/3_logs/xkalman_25.txt
python test_model.py const_v x 50 all > ../data/3_logs/const_v_50.txt
python test_model.py const_a x 50 all > ../data/3_logs/const_a_50.txt
python test_model.py kinematics x 50 all > ../data/3_logs/kinematics_50.txt
python test_model.py xkalman x 50 all > ../data/3_logs/xkalman_50.txt
python test_model.py const_v x 75 all > ../data/3_logs/const_v_75.txt
python test_model.py const_a x 75 all > ../data/3_logs/const_a_75.txt
python test_model.py kinematics x 75 all > ../data/3_logs/kinematics_75.txt
python test_model.py xkalman x 75 all > ../data/3_logs/xkalman_75.txt
python test_model.py const_v x 100 all > ../data/3_logs/const_v_100.txt
python test_model.py const_a x 100 all > ../data/3_logs/const_a_100.txt
python test_model.py kinematics x 100 all > ../data/3_logs/kinematics_100.txt
python test_model.py xkalman x 100 all > ../data/3_logs/xkalman_100.txt
```

#### 2) Create Log Files Of Machine Learning Based Models (via Training)
[...]

#### 3) Script To Merge All Logs and Create Performance Table
```
python log_parser_classic.py const_v
python log_parser_classic.py const_a
python log_parser_classic.py kinematics
python log_parser_classic.py xkalman
python log_parser_ml.py social_lstm
python log_parser_ml.py social_bigats
python log_parser_ml.py physics_lstm
python log_parser_ml.py gatsbiv1
python log_parser_ml.py gatsbiv2
```

## [License](#license)
This repository will be published on GitHub upon publication at Neurips25 under the MIT license.
For further details, please find the **LICENSE** file in this repository.



## [Cluster & Runtime](#cluster)

We used our university's computational facility that provided a Linux cluster (OS: Ubuntu 22.04.5 LTS, Kernel: Linux 5.15.0-134-generic) with the Slurm workload manager and GPUs. CUDA (3.11.6_cuda) and Python (v3.11.6) were installed.

In the following we outline several linux commands that we used to automate training and testing.

**[!!!] Important Note:** All of the following commands are executed from within folder `./neurips25_great_gatsbi/src/`.

```
cd ./neurips25_great_gatsbi/src/
```

### 1. Prepare Training Dataset
(takes time!)

### 2. Train Model
(takes around 3h)

For each model (social_lstm) and prediction_length (25, 50, 75, 100) we run ten epochs, that take around 2h.
We repeated the same 5 times, so the training was 5 times for 10 epochs each in the order the data appears in the script below.

```
./_submit_jobs.sh social_lstm 25 10
```

```
#!/bin/bash

# Usage: ./_submit_jobs.sh <model_name> <prediction_length> <multimodal> <num_jobs_per_split>
if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_name> <prediction_length> <multimodal> <num_jobs_per_split>"
    exit 1
fi

MODEL_NAME=$1
PRED_LEN=$2
MULTI_MODAL=$3
NUM_JOBS=$4

SPLITS=(split_1 split_2 split_3 split_4 split_5)

echo "The following job submission commands will be executed:"
for SPLIT in "${SPLITS[@]}"; do
    echo "Processing $SPLIT:"
    for i in $(seq 1 $NUM_JOBS); do
        if [ $i -eq 1 ]; then
            echo "  sbatch -n4 -G 2 --time=02:30:00 --gres=gpumem:10g --mem-per-cpu=8000 --wrap=\"module load stack/2024-05 python/3.11.6_cuda ; python train_model.py $MODEL_NAME $PRED_LEN 50 $SPLIT $MULTI_MODAL\""
        else
            echo "  sbatch --dependency=afterok:<jobid_${SPLIT}_$((i-1))> -n4 -G 2 --time=02:30:00 --gres=gpumem:10g --mem-per-cpu=8000 --wrap=\"module load stack/2024-05 python/3.11.6_cuda ; python train_model.py $MODEL_NAME $PRED_LEN 50 $SPLIT $MULTI_MODAL\""
        fi
    done
done

echo
read -p "Press Enter to confirm and submit the jobs..."

# Actual submission with dependency chaining per split
for SPLIT in "${SPLITS[@]}"; do
    PREV_JOBID=""
    echo "Submitting jobs for $SPLIT..."
    for i in $(seq 1 $NUM_JOBS); do
        if [ -z "$PREV_JOBID" ]; then
            JOBID=$(sbatch --parsable -n4 -G 2 --time=02:30:00 --gres=gpumem:10g --mem-per-cpu=8000 \
                --wrap="module load stack/2024-05 python/3.11.6_cuda ; python train_model.py $MODEL_NAME $PRED_LEN 50 $SPLIT $MULTI_MODAL")
        else
            JOBID=$(sbatch --parsable --dependency=afterok:$PREV_JOBID -n4 -G 2 --time=02:30:00 --gres=gpumem:10g --mem-per-cpu=8000 \
                --wrap="module load stack/2024-05 python/3.11.6_cuda ; python train_model.py $MODEL_NAME $PRED_LEN 50 $SPLIT $MULTI_MODAL")
        fi
        echo "  Submitted job $JOBID (iteration $i for $SPLIT)"
        PREV_JOBID=$JOBID
    done
done
```

### 3. Test Model (outdated)
(takes around 1 minute)

The following script parses all summaries from models folder to assess for which epoch, prediction_length, and model the performance on testing set was best.
```
#!/bin/bash

DATA_DIR="$HOME/neurips25_great_gatsbi/data/4_models"

for model_name in social_lstm physics_lstm; do
  for prediction_length in 25 50 75 100; do
    for epoch in $(seq -w 0 49); do
      file_path="${DATA_DIR}/${model_name}_${prediction_length}_${epoch}.model_perf.txt"
      if [ -f "$file_path" ]; then
        file_content=$(tr '\n' ' ' < "$file_path" | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
      else
        file_content="File not found"
      fi
      echo -e "${model_name}\t${prediction_length}\t${epoch}\t${file_content}"
    done
  done
done
```