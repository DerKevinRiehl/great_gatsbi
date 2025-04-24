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

| sequence_nr | video_file                    | part    | from_frame | to_frame | num_bicycles |
|-------------|-------------------------------|---------|------------|----------|--------------|
| 1           | DJI_20240906103036_0003_D.MP4 | PART_1  | 300        | 1950     | 6            |
| 2           | DJI_20240906103036_0003_D.MP4 | PART_2  | 2425       | 3450     | 10           |
| 3           | DJI_20240906103036_0003_D.MP4 | PART_3  | 5200       | 5350     | 10           |
| 4           | DJI_20240906103036_0003_D.MP4 | PART_4  | 5625       | 6154     | 14           |
| 5           | DJI_20240906103442_0004_D.MP4 | PART_1  | 0          | 1375     | 14           |
| 6           | DJI_20240906103442_0004_D.MP4 | PART_2  | 2850       | 4500     | 19           |
| 7           | DJI_20240906103850_0005_D.MP4 | PART_1  | 325        | 2050     | 22           |
| 8           | DJI_20240906105321_0009_D.MP4 | PART_1  | 150        | 350      | 13           |
| 9           | DJI_20240906105621_0010_D.MP4 | PART_1  | 350        | 925      | 6            |
| 10          | DJI_20240906105621_0010_D.MP4 | PART_2  | 1250       | 1900     | 9            |
| 11          | DJI_20240906105621_0010_D.MP4 | PART_3  | 2250       | 2875     | 12           |
| 12          | DJI_20240906105621_0010_D.MP4 | PART_4  | 3075       | 3250     | 16           |
| 13          | DJI_20240906105621_0010_D.MP4 | PART_5  | 3250       | 3700     | 17           |
| 14          | DJI_20240906105621_0010_D.MP4 | PART_6  | 5950       | 6138     | 17           |
| 15          | DJI_20240906110027_0011_D.MP4 | PART_1  | 0          | 1725     | 17           |
| 16          | DJI_20240906110027_0011_D.MP4 | PART_2  | 2525       | 3300     | 17           |
| 17          | DJI_20240906110027_0011_D.MP4 | PART_3  | 3500       | 4375     | 17           |
| 18          | DJI_20240906110027_0011_D.MP4 | PART_4  | 4675       | 5500     | 17           |
| 19          | DJI_20240906110027_0011_D.MP4 | PART_5  | 5850       | 6122     | 17           |
| 20          | DJI_20240906110432_0012_D.MP4 | PART_1  | 0          | 625      | 17           |

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
We recommend to **precalculate all training data from the trajectory data**, as this is time consuming.
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

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## [License](#license)
This repository will be published on GitHub upon publication at Neurips25 under the MIT license.
For further details, please find the **LICENSE** file in this repository.