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
| 20          | DJI_20240906110432_0012_D.MP4 | PART_1  | 0          | 625      | 625        | 17           | test  |
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

### Prepratation of Trajectory Dataset
Please extract all trajectory txt files from `\neurips25_great_gatsbi\data\1_trajectories\1_trajectories.zip` and store them in the folder `\neurips25_great_gatsbi\data\1_trajectories`.

On Linux you could use this command:
```
unzip neurips25_great_gatsbi/data/1_trajectories/1_trajectories.zip -d neurips25_great_gatsbi/data/1_trajectories/
```

## [Training](#training)

### Training Data Generation
We recommend to **precalculate all training data from the trajectory data**, as this is time consuming and takes up to 20 minutes.
First, training data needs to be generated with the script `data_generator.py`. 
The results are stored in `\neurips25_great_gatsbi\data\2_training_datasets`.

The script can be used as follows:
```
python data_generator.py [1] [2] [3] [4]
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi")
    [4] - data_type ("train" or "test")
```

An example to generate data can be found here:

```
python data_generator.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm train
```

A list of all training relevant sequences is outlined below:
<details>
    All sequences related to these videos have been used for training.
    <ul>
        <li> DJI_20240906103036_0003_D.MP4 </li>
        <li> DJI_20240906103442_0004_D.MP4 </li>
        <li> DJI_20240906103850_0005_D.MP4 </li>
        <li> DJI_20240906105321_0009_D.MP4 </li>
        <li> DJI_20240906105621_0010_D.MP4 </li>
    </ul>
</details>

### Model Training

To train the model you can use  the script `train_model.py`. 
The resulting models are stored in `\neurips25_great_gatsbi\data\4_models`.

The script can be used as follows:
```
python train_model.py [1] [2] [3] [4]")
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi")
    [4] - prediction_length in [s] (25, 50 , 75, 100)
    [5] - n_epochs   
```

An example to train a model can be found here:
```
python train_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm 25 10
```

After training on a specific sequence (video+part) two model files are stored in the models folder:
- general workbench model (Example: *social_lstm_25_5.model*)
    - if this model file exists in the model folder already, then it will be loaded, and continued to be trained
- specific snapshot model (Example: *social_lstm_25_5_DJI_20240906103036_0003_D.MP4-PART_3.model*)
    - this is a copy of a workbench that is the result after training on a specific sequence

This storage allows to retrieve models of different training amounts.


## [Evaluation](#evaluation)

### Testing Data Generation

We recommend to **precalculate all testing data from the trajectory data**, as this is time consuming.
First, testing data needs to be generated with the script `data_generator.py`. 
The results are stored in `\neurips25_great_gatsbi\data\3_testing_datasets`.

The script can be used as follows:
```
python data_generator.py [1] [2] [3] [4]
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi")
    [4] - data_type ("train" or "test")
```

An example to generate data can be found here:

```
python data_generator.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm test
```

A list of all testing / evaluation relevant sequences is outlined below:
<details>
    All sequences related to these videos have been used for training.
    <ul>
        <li> DJI_20240906110027_0011_D.MP4</li>
        <li> DJI_20240906110432_0012_D.MP4 </li>
    </ul>
</details>

### Evaluation

To test the model (and thus evaluate) you can use the script `test_model.py`. 
The resulting evaluation metrics are printed to the console.

The script can be used as follows:
```eval
python test_model.py [1] [2] [3] [4]")
    [1] - relevant_video
    [2] - relevant_part
    [3] - model ("social_lstm" or "gatsbi" or "const_v" or "const_a")
    [4] - model_file_name
    [5] - prediction_length in [s] (25, 50 , 75, 100)
```

An example to run a test can be found here:
```
python test_model.py DJI_20240906103036_0003_D.MP4 PART_3 social_lstm social_lstm_25_5_0010.model 25
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
The benchmark of different models shows that the proposed GATsBi model is outperforming pedestrian specific (social_lstm) and car specific (const_v, const_a) models.

| Model  | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |
|------------|----|----|----|----|----|----|----|----|
| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |
| **car specific (physics)** |   |   |   |   |  |   |   |   |
| const_v    | 0.1106 | 0.2923 | 0.5589 | 0.9452 | 0.2672 | 0.6844 | 1.5290 | 2.6763 |
| const_a    | 0.1270 | 0.5491 | 1.2874 | 2.3715 | 0.3914 | 1.6356 | 3.9722 | 7.3009 |
| kinematics | 0.1113 | 0.3972 | 0.8901 | 1.6178 | 0.3051 | 1.1105 | 2.6968 | 4.9086 |
| xkalman    | 0.1489 | 0.3404 | 0.6141 | 1.0054 | 0.3179 | 0.7489 | 1.6019 | 2.7542 |
| **pedestrian specific** |   |   |   |   |  |   |   |   |
| social_lstm (10ep) | 0.2051 | 0.3899 | 0.7601 | 4.1339 | 0.4583 | 0.7872 | 1.8785 | 7.9684 |
| social_lstm (20ep) | 0.1923 | 1.9488 | 3.1585 | 4.2578 | 0.4207 | 3.5531 | 6.0151 | 7.6663 |

### All digits... [to be deleted]
| Model | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |
|------------|----|----|----|----|----|----|----|----|
| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |
| **car specific** |   |   |   |   |  |   |   |   |
| const_v    | 0.11056696657038025 | 0.29234241037736436 | 0.5589889535979709 | 0.945163672885311 |  0.2671894703293539  | 0.6844184623060213 | 1.5290085477690523 | 2.6763083424922742 |
| const_a    | 0.12703750214115742 | 0.5491118972058714 | 1.2873844576742641 | 2.371540486730733 | 0.3913595031669894 | 1.635600893058238 | 3.972229972595745 | 7.300925779744927 |
| kinematics | 0.11131377199367609 | 0.39723217019572643 | 0.8900938109835153 | 1.6178176029537246 | 0.30509164684984014 | 1.1104835422805694 | 2.696847655526652 | 4.908569119363433 |
| xkalman | 0.1489094605288226 | 0.34044283121526064 | 0.6141094498969256 | 1.0053677485343262 | 0.317884474618542 | 0.748918094949814 | 1.6018550991895084 | 2.754164592130338 |
| **pedestrian specific** |   |   |   |   |  |   |   |   |
| social_lstm (10ep) | 0.20509553169628814 | 0.3899093248726271 | 0.7601036562366721 | 4.133902388973659 | 0.45827985002242844 | 0.7872095224037342 | 1.8784515377772142 | 7.968354260334436 |
| social_lstm (20ep) |  0.19227776936463148 | 1.9488191385935258 | 3.1585467262129607 | 4.257847008154115 | 0.42069589445023575 | 3.5531278204305137 | 6.01511868088999 | 7.666314679920989 |




>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


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
(takes around 30 minutes)
```
#!/bin/bash

# Configurable parameters
model="social_lstm" # Set your desired model name # "social_lstm", "gatsbi", "physics_lstm"

# List of video files and their corresponding parts
declare -A FILE_TRAIN
FILE_TRAIN["DJI_20240906103036_0003_D.MP4"]="PART_1 PART_2 PART_3 PART_4"
FILE_TRAIN["DJI_20240906103442_0004_D.MP4"]="PART_1 PART_2"
FILE_TRAIN["DJI_20240906103850_0005_D.MP4"]="PART_1"
FILE_TRAIN["DJI_20240906105321_0009_D.MP4"]="PART_1"
FILE_TRAIN["DJI_20240906105621_0010_D.MP4"]="PART_1 PART_2 PART_3 PART_4 PART_5 PART_6"
declare -A FILE_TEST
FILE_TEST["DJI_20240906110027_0011_D.MP4"]="PART_1 PART_2 PART_3 PART_4 PART_5"
FILE_TEST["DJI_20240906110432_0012_D.MP4"]="PART_1"

# SLURM parameters
SBATCH_OPTS="-n4 --time=02:00:00 --mem-per-cpu=16000"
MODULES="module load stack/2024-05 python/3.11.6_cuda"
for FILE in "${!FILE_TRAIN[@]}"; do
    for PART in ${FILE_TRAIN[$FILE]}; do
        CMD="$MODULES ; python data_generator.py $FILE $PART $model train"
        sbatch $SBATCH_OPTS --wrap="$CMD"
    done
done
SBATCH_OPTS="-n4 --time=02:00:00 --mem-per-cpu=16000"
MODULES="module load stack/2024-05 python/3.11.6_cuda"
for FILE in "${!FILE_TEST[@]}"; do
    for PART in ${FILE_TEST[$FILE]}; do
        CMD="$MODULES ; python data_generator.py $FILE $PART $model test"
        sbatch $SBATCH_OPTS --wrap="$CMD"
    done
done
```

**[!!!] Important Note:** Verify that all files exist before you go to the next step with:
```
all_exist=true
missing_files=()

# Check training files
for file in "${!FILE_TRAIN[@]}"; do
    for part in ${FILE_TRAIN[$file]}; do
        fname="data_social_lstm_${file}-${part}.pt"
        fpath="../data/2_training_datasets/$fname"
        if [[ ! -f "$fpath" ]]; then
            all_exist=false
            missing_files+=("$fpath")
        fi
    done
done

# Check testing files
for file in "${!FILE_TEST[@]}"; do
    for part in ${FILE_TEST[$file]}; do
        fname="data_social_lstm_${file}-${part}.pt"
        fpath="../data/3_testing_datasets/$fname"
        if [[ ! -f "$fpath" ]]; then
            all_exist=false
            missing_files+=("$fpath")
        fi
    done
done

if $all_exist; then
    echo "[!!!] yes all files existing! ready for training!"
else
    echo "[!!!] no some files missing! not ready for training yet!"
    echo "Missing files:"
    for f in "${missing_files[@]}"; do
        echo "$f"
    done
fi
```
### 2. Train Model
(takes around 8h)

For each model (social_lstm) and prediction_length (25, 50, 75, 100) we run ten epochs, that take around 8h.
We repeated the same 5 times, so the training was 5 times for 10 epochs each in the order the data appears in the script below.
```
#!/bin/bash

# Configurable parameters 
prediction_length=25   # Set your desired prediction length # "25", "50", "75", "100"
model="social_lstm"    # Set your desired model name # "social_lstm", "gatsbi", "physics_lstm"
n_epochs=10            # Set your desired number of epochs

# Activate software environment for session host as well
module load stack/2024-05 python/3.11.6_cuda

# SLURM resource options
SBATCH_OPTS="-n4 -G2 --time=01:30:00 --gres=gpumem:10g --mem-per-cpu=8000"
MODULES="module load stack/2024-05 python/3.11.6_cuda"

# Map video files to their parts
declare -A FILE_PARTS
FILE_PARTS["DJI_20240906103036_0003_D.MP4"]="PART_1 PART_2 PART_3 PART_4"
FILE_PARTS["DJI_20240906103442_0004_D.MP4"]="PART_1 PART_2"
FILE_PARTS["DJI_20240906103850_0005_D.MP4"]="PART_1"
FILE_PARTS["DJI_20240906105321_0009_D.MP4"]="PART_1"
FILE_PARTS["DJI_20240906105621_0010_D.MP4"]="PART_1 PART_2 PART_3 PART_4 PART_5 PART_6"

# Build the job list
jobs=()
for file in "${!FILE_PARTS[@]}"; do
    for part in ${FILE_PARTS[$file]}; do
        jobs+=("python train_model.py $file $part $model $prediction_length $n_epochs")
    done
done
printf '%s\n' "${jobs[@]}"

# Wait for user to press enter to confirm
read -p "Please enter to submit jobs"

# Submit jobs with dependencies
jobid=""
for i in "${!jobs[@]}"; do
    cmd="$MODULES ; date ; ${jobs[$i]} ; date"
    if [[ $i -eq 0 ]]; then
        jobid=$(sbatch $SBATCH_OPTS --wrap="$cmd" | awk '{print $4}')
    else
        jobid=$(sbatch --dependency=afterok:$jobid $SBATCH_OPTS --wrap="$cmd" | awk '{print $4}')
    fi
done
```

### 3. Test Model
(takes around 5 minutes)
```
#!/bin/bash

# Configurable parameters 
prediction_length=25   # Set your desired prediction length # "25", "50", "75", "100"
model="social_lstm"    # Set your desired model name # "social_lstm", "gatsbi", "physics_lstm"
model_file=""

# Map video files to their parts
declare -A FILE_TEST
FILE_TEST["DJI_20240906110027_0011_D.MP4"]="PART_1 PART_2 PART_3 PART_4 PART_5"
FILE_TEST["DJI_20240906110432_0012_D.MP4"]="PART_1"

# Build the job list
jobs=()
for file in "${!FILE_TEST[@]}"; do
    for part in ${FILE_TEST[$file]}; do
        jobs+=("python test_model.py $file $part $model $model_file $prediction_length")
    done
done

# Submit jobs with dependencies
for i in "${!jobs[@]}"; do
    cmd="$MODULES ; date ; ${jobs[$i]} ; date"
    eval "$cmd"
done
```