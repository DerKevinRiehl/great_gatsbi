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
| **conventional (physics)** |   |   |   |   |   |   |   |   |
| const_v | 0.1080 | 0.2818 | 0.5460 | 0.9406 | 0.2592 | 0.6568 | 1.5245 | 2.7275 |
|       |[0.0076]|[0.0194]|[0.0444]|[0.1059]|[0.0182]|[0.0436]|[0.1787]|[0.4278]|
| const_a | 0.1281 | 0.5504 | 1.2951 | 2.3929 | 0.3934 | 1.6373 | 4.0117 | 7.3837 |
|       |[0.0118]|[0.0482]|[0.1180]|[0.2292]|[0.0346]|[0.1422]|[0.3857]|[0.7451]|
| kinematics | 0.1103 | 0.3942 | 0.8914 | 1.6309 | 0.3027 | 1.1047 | 2.7238 | 4.9800 |
|       |[0.0088]|[0.0364]|[0.0905]|[0.1795]|[0.0260]|[0.1068]|[0.3056]|[0.5935]|
| xkalman | 0.1445 | 0.3269 | 0.5967 | 0.9948 | 0.3068 | 0.7154 | 1.5887 | 2.7913 |
|       |[0.0122]|[0.0242]|[0.0512]|[0.1146]|[0.0235]|[0.0492]|[0.1904]|[0.4417]|
| **new run mchine learning (unimodal)** |   |   |   |   |   |   |   |   |
| ego_lstm | 0.0710 | 0.2158 | 0.4450 | 0.7900 | 0.1891 | 0.5038 | 1.2597 | 2.3289 |
|       | [0.0062] | [0.0166] | [0.0387] | [0.1036] | [0.0148] | [0.0358] | [0.1673] | [0.4284] |
| social_lstm | 0.0876 | 0.2487 | 0.4762 | 0.8214 | 0.2141 | 0.5479 | 1.2829 | 2.3770 |
|       | [0.0071] | [0.0133] | [0.0359] | [0.0911] | [0.0162] | [0.0332] | [0.1674] | [0.4008] |
| social_bigat | 0.0774 | 0.2315 | 0.4708 | 0.8211 | 0.1984 | 0.5209 | 1.2938 | 2.3679 |
|       | [0.0062] | [0.0138] | [0.0387] | [0.0856] | [0.0146] | [0.0250] | [0.1536] | [0.4044] |
| physics_lstm | 0.0752 | 0.2158 | 0.4423 | 0.7943 | 0.1973 | 0.5046 | 1.2667 | 2.3568 |
|       | [0.0062] | [0.0165] | [0.0431] | [0.1015] | [0.0156] | [0.0340] | [0.1784] | [0.4198] |
| physics_lstmv2 (no ego) | 0.0810 | 0.2288 | 0.4489 | 0.7959 | 0.2111 | 0.5302 | 1.2786 | 2.3619 |
|       | [0.0052] | [0.0141] | [0.0371] | [0.1005] | [0.0138] | [0.0303] | [0.1671] | [0.4194] |
| gatsbiv1 | 0.0763 | 0.2181 | 0.4222 | 0.7812 | 0.1955 | 0.4996 | 1.2218 | 2.3137 |
|       | [0.0031] | [0.0167] | [0.0342] | [0.1045] | [0.0091] | [0.0264] | [0.1548] | [0.4081] |
| gatsbiv2 | 0.0771 | 0.2140 | 0.4245 | 0.7633 | 0.1970 | 0.4936 | 1.2259 | 2.3050 |
|       | [0.0043] | [0.0134] | [0.0392] | [0.0865] | [0.0100] | [0.0290] | [0.1695] | [0.3786] |
| gatsbiv4 | 0.0750 | 0.2114 | 0.4338 | 0.7705 | 0.1939 | 0.4897 | 1.2321 | 2.3016 |
|       | [0.0062] | [0.0129] | [0.0338] | [0.0880] | [0.0145] | [0.0280] | [0.1527] | [0.4015] |
| gatsbiv4_physics_ablation | 0.0754 | 0.2269 | 0.4375 | 0.7741 | 0.1951 | 0.5023 | 1.2322 | 2.2813 |
|       | [0.0080] | [0.0144] | [0.0357] | [0.0910] | [0.0175] | [0.0280] | [0.1577] | [0.4127] |
| gatsbiv5 | 0.0755 | 0.2147 | 0.4324 | 0.7758 | 0.1942 | 0.4942 | 1.2302 | 2.2876 |
|       | [0.0063] | [0.0114] | [0.0382] | [0.1016] | [0.0150] | [0.0239] | [0.1683] | [0.4270] |
| gatsbiv6 | 0.0750 | 0.2152 | 0.4333 | 0.7806 | 0.1922 | 0.4952 | 1.2452 | 2.3036 |
|       | [0.0059] | [0.0075] | [0.0329] | [0.1024] | [0.0136] | [0.0179] | [0.1539] | [0.4275] |
| **machine learning NEW RUN (multimodal_gmm, expected only)** |   |   |   |   |   |   |   |   |
| ego_lstm | 0.0628 | 0.2043 | 0.4275 | 0.7946 | 0.1771 | 0.4913 | 1.2635 | 2.4709 |
|   | [0.0057] | [0.0196] | [0.0392] | [0.0902] | [0.0145] | [0.0403] | [0.1695] | [0.3747] |
| social_lstm  | 0.0820 | 0.2341 | 0.4630 | 0.8314 | 0.2077 | 0.5393 | 1.3235 | 2.5515 |
|  | [0.0090] | [0.0172] | [0.0202] | [0.0929] | [0.0191] | [0.0360] | [0.1601] | [0.3610] |
|  social_bigat  | 0.0702 | 0.2240 | 0.4586 | 0.8069 | 0.1914 | 0.5242 | 1.3234 | 2.5356 |
|     | [0.0068] | [0.0139] | [0.0377] | [0.0898] | [0.0138] | [0.0304] | [0.1302] | [0.3435] |
|  physics_lstm | 0.0665 | 0.2130 | 0.4319 | 0.7821 | 0.1839 | 0.5060 | 1.3187 | 2.4470 |
|     | [0.0060] | [0.0130] | [0.0427] | [0.0919] | [0.0139] | [0.0272] | [0.1703] | [0.4019] |
| physics_lstmv2 (no ego) | 0.0802 | 0.2263 | 0.4513 | 0.8045 | 0.2110 | 0.5335 | 1.3292 | 2.4936 |
|   | [0.0057] | [0.0140] | [0.0365] | [0.0924] | [0.0136] | [0.0313] | [0.1714] | [0.3703] |
| gatsbiv4 | 0.0698 | 0.2107 | 0.4216 | 0.7768 | 0.1869 | 0.4990 | 1.3048 | 2.5425 |
|  | [0.0059] | [0.0137] | [0.0328] | [0.0961] | [0.0131] | [0.0346] | [0.1425] | [0.4363] |
| gatsbiv4_physics_ablation | 0.0642 | 0.2135 | 0.4323 | 0.7834 | 0.1765 | 0.4963 | 1.2964 | 2.4370 |
|  | [0.0063] | [0.0139] | [0.0302] | [0.0918] | [0.0139] | [0.0297] | [0.1553] | [0.3308] |
| gatsbiv5 | 0.0662 | 0.2079 | 0.4212 | 0.7761 | 0.1816 | 0.4887 | 1.2968 | 2.4916 |
|  | [0.0089] | [0.0103] | [0.0325] | [0.0893] | [0.0186] | [0.0254] | [0.1315] | [0.4005] |

| Model  | ADE | ADE | ADE | ADE | FDE | FDE | FDE | FDE |
|------------|----|----|----|----|----|----|----|----|
| *prediction length*           | *1s* | *2s* | *3s* | *4s* | *1s* | *2s* | *3s* | *4s* |
| **machine learning NEW RUN (multimodal_gmm)** |   |   |   |   |   |   |   |   |
| social_lstm |   |   |   |   |   |   |   |   |
| *_____(best)* | 0.0703 | 0.2123 | 0.4365 | 0.7291 | 0.1782 | 0.4949 | 1.1216 | 1.8998 |
|   | [0.0074] | [0.0135] | [0.0188] | [0.0592] | [0.0162] | [0.0564] | [0.0994] | [0.2921] |
| *_____(most prob)* | 0.0848 | 0.2705 | 0.5304 | 1.0739 | 0.2093 | 0.5458 | 1.3133 | 2.3634 |
|   | [0.0098] | [0.0247] | [0.0278] | [0.1759] | [0.0208] | [0.0362] | [0.1111] | [0.4013] |
| *_____(sampled)* | 0.1114 | 0.5974 | 1.6203 | 5.4131 | 0.3211 | 1.8832 | 5.4571 | 12.4895 |
|     | [0.0066] | [0.4137] | [0.5398] | [1.4416] | [0.0822] | [1.4303] | [2.0081] | [1.7183] |
| *_____(expected)*   | 0.0820 | 0.2341 | 0.4630 | 0.8314 | 0.2077 | 0.5393 | 1.3235 | 2.5515 |
|  | [0.0090] | [0.0172] | [0.0202] | [0.0929] | [0.0191] | [0.0360] | [0.1601] | [0.3610] |
| social_bigat |   |   |   |   |   |   |   |   |
| *_____(best)* | 0.0602 | 0.2146 | 0.4569 | 0.6828 | 0.1655 | 0.4856 | 1.1771 | 1.6954 |
|   | [0.0052] | [0.0163] | [0.0590] | [0.0676] | [0.0104] | [0.0268] | [0.1894] | [0.2033] |
| *_____(most prob)* | 0.0749 | 0.2526 | 0.5342 | 0.9819 | 0.1917 | 0.5254 | 1.2852 | 2.3769 |
|   | [0.0065] | [0.0193] | [0.1166] | [0.1055] | [0.0143] | [0.0287] | [0.1660] | [0.4126] |
|  *_____(sampled)* | 0.2194 | 1.1598 | 1.6950 | 6.2134 | 1.3577 | 3.6539 | 5.8660 | 13.6125 |
|    | [0.1633] | [0.5325] | [0.5229] | [1.1436] | [1.6154] | [1.3359] | [1.8090] | [1.4614] |
|  *_____(expected)*   | 0.0702 | 0.2240 | 0.4586 | 0.8069 | 0.1914 | 0.5242 | 1.3234 | 2.5356 |
|     | [0.0068] | [0.0139] | [0.0377] | [0.0898] | [0.0138] | [0.0304] | [0.1302] | [0.3435] |
| physics_lstm |   |   |   |   |   |   |   |   |
| *_____(best)* | 0.0555 | 0.2565 | 0.4004 | 0.6765 | 0.1530 | 0.6977 | 1.0327 | 1.5322 |
|    | [0.0055] | [0.0516] | [0.0085] | [0.0555] | [0.0153] | [0.1308] | [0.0800] | [0.1780] |
|  *_____(most prob)* | 0.0688 | 0.5305 | 0.5244 | 0.8485 | 0.1857 | 0.5065 | 1.2684 | 2.3344 |
|    | [0.0062] | [0.1894] | [0.0691] | [0.0629] | [0.0138] | [0.0281] | [0.1948] | [0.4310] |
|  *_____(sampled)* | 158706.2950 | 1.1474 | 2.5434 | 6.6921 | 3136507.2398 | 3.8130 | 7.5892 | 14.4773 |
|     | [317382.0409] | [0.5870] | [0.8111] | [0.1273] | [6273013.3801] | [1.5864] | [1.2849] | [0.4441] |
|   *_____(expected)*  | 0.0665 | 0.2130 | 0.4319 | 0.7821 | 0.1839 | 0.5060 | 1.3187 | 2.4470 |
|     | [0.0060] | [0.0130] | [0.0427] | [0.0919] | [0.0139] | [0.0272] | [0.1703] | [0.4019] |
| gatsbiv4_physics_ablation |   |   |   |   |   |   |   |   |
| *_____(best)* | 0.0557 | 0.2053 | 0.4062 | 0.7263 | 0.1515 | 0.4509 | 1.0214 | 1.9168 |
|               | [0.0047] | [0.0127] | [0.0327] | [0.0957] | [0.0108] | [0.0199] | [0.1036] | [0.3188] |
| *_____(most prob)* | 0.0683 | 0.2311 | 0.4914 | 1.2600 | 0.1777 | 0.4960 | 1.2291 | 2.3218 |
|                    | [0.0063] | [0.0141] | [0.0646] | [0.4419] | [0.0139] | [0.0304] | [0.1635] | [0.4249] |
| *_____(sampled)* | 0.3220 | 1.1455 | 2.9130 | 4.4423 | 0.4858 | 3.6718 | 8.0018 | 10.8022 |
|                  | [0.4169] | [0.5892] | [1.3024] | [0.2033] | [0.2248] | [1.4568] | [2.2389] | [0.2663] |
| *_____(expected)* | 0.0642 | 0.2135 | 0.4323 | 0.7834 | 0.1765 | 0.4963 | 1.2964 | 2.4370 |
|                   | [0.0063] | [0.0139] | [0.0302] | [0.0918] | [0.0139] | [0.0297] | [0.1553] | [0.3308] |
| gatsbiv5 |   |   |   |   |   |   |   |   |
| *_____(best)* | 0.0564 | 0.2724 | 0.5451 | 0.7360 | 0.1546 | 0.5891 | 1.2407 | 1.7462 |
|               | [0.0064] | [0.0557] | [0.0623] | [0.0985] | [0.0150] | [0.0598] | [0.1666] | [0.1864] |
| *_____(most prob)* | 0.0693 | 0.3600 | 0.8161 | 1.2315 | 0.1822 | 0.5463 | 1.4792 | 2.4727 |
|                    | [0.0099] | [0.0768] | [0.1659] | [0.3759] | [0.0190] | [0.0203] | [0.1201] | [0.3896] |
| *_____(sampled)* | 0.1318 | 1.2634 | 2.6584 | 6.4608 | 0.5425 | 4.2846 | 7.0888 | 14.1942 |
|                  | [0.0404] | [0.3167] | [1.7631] | [0.5566] | [0.1733] | [0.8106] | [3.5614] | [0.7392] |
| *_____(expected)*  | 0.0662 | 0.2079 | 0.4212 | 0.7761 | 0.1816 | 0.4887 | 1.2968 | 2.4916 |
|                    | [0.0089] | [0.0103] | [0.0325] | [0.0893] | [0.0186] | [0.0254] | [0.1315] | [0.4005] |





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