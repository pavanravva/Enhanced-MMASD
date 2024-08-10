# Introduction

In this repository, we present a Multimodal Privacy-Preserving Dataset focused on play therapy interventions for children with ASD, along with a Multimodal Transformer Framework. The dataset comprises three types of data: 3D skeleton data, 3D body mesh data, and optical flow data, all extracted from video recordings of play therapy sessions between therapists and children with ASD. The framework is designed for two key tasks: action classification and ASD identification.

The following table lists the different actions performed during therapy sessions along with the number of videos available for each action class:

| S.NO | Action Class             | No. Videos |
|:----:|:-------------------------|-----------:|
| 1    | Arm Swing                | 105        |
| 2    | Body Swing               | 119        |
| 3    | Chest Expansion          | 114        |
| 4    | Drumming                 | 168        |
| 5    | Sing and Clap            | 113        |
| 6    | Twist Pose               | 120        |
| 7    | Tree Pose                | 129        |
| 8    | Frog Pose                | 113        |
| 9    | Squat Pose               | 101        |
| 10   | Maracas Forward Shaking  | 103        |
| 11   | Maracas Shaking          | 130        |

**Table 1:** Action Class and their size


# Privacy Features Exraction:
Before extracting the provacy features, the raw video recording undergo some pre-processing steps, the individual persons are extracted form the video, and then each person privacy features are extracted. The detailes steps of features extraction can be illustrated in Figure 1. Complete data will be released after paper been accepted, for now we are uploading few samples of privacy features for each modality.

<div align="center">
  <img src="data_flow_chart_mmasd.jpg" alt="Flowchart of Data Processing (don't forget to change ROMP image)" width="80%">

  <p><strong>Figure 1:</strong> Flowchart of Data Processing</p>
</div>

# Multimodal Tranformer Framework

For evaluating the power of the dataset, we have first tested it on two independent tasks action classification, and identifying ASD. For tasks we have tested on multiple combination of dataset and frameworks as listed in **Table 2**. The results for Action Classification, and ASD identification are listed in **Table 3** and **Table 4**

| S.NO | Data Combination         | Name | ML Frameworks           |
|------|--------------------------|------|-------------------------|
| 1    | SCD                      | A1   | LSTM                    |
| 2    | RVD                      | A2   | CNN, ViViT              |
| 3    | OFD                      | A3   | CNN, ViViT              |
| 4    | SCD + RV                 | A4   | LSTM + CNN              |
| 5    | SCD + OF                 | A5   | LSTM + CNN              |
| 6    | RVD + OFD                | A6   | CNN + CNN, ViViT + CNN  |
| 7    | SCD + RVD + OFD          | A7   | LSTM + CNN + CNN, LSTM + ViViT + CNN |

**Table 2:** Data Modalities and ML Frameworks

**Notes:**
- **SCD**: 3D Skeleton Coordinate Data
- **RVD**: ROMP Video Data
- **OFD**: Optical Flow Data


| S.No | Data-Model Combination                               | Accuracy | F1    |
|------|------------------------------------------------------|----------|-------|
| 1    | **A1**: SCD → LSTM                                   | 0.8237   | 0.81  |
| 2    | **A2**: RVD → CNN                                    | 0.8247   | 0.8245|
| 3    | **A2**: RVD → ViViT                                  | 0.8976   | 0.8836|
| 4    | **A3**: OFD → CNN                                    | 0.86     | 0.8558|
| 5    | **A3**: OFD → ViViT                                  | 0.9024   | 0.8924|
| 6    | **A4**: SCD → LSTM, RVD → CNN                        | 0.937    | 0.94  |
| 7    | **A5**: SCD → LSTM, OFD → CNN                        | 0.9678   | 0.956 |
| 8    | **A6**: RVD → CNN, OFD → CNN                         | 0.9671   | 0.96  |
| 9    | **A6**: RVD → ViViT, OFD → CNN                       | 0.9689   | 0.9687|
| 10   | **A6**: RVD → CNN, OFD → ViViT                       | 0.9505   | 0.9502|
| 11   | **A7**: RVD → ViViT, OFD → CNN, SCD → LSTM           | **0.968**| **0.97** |
| 12   | **A7**: RVD → CNN, OFD → ViViT, SCD → LSTM           | 0.9564   | 0.95  |

**Table 2:** Action Classification Results of Various Data-Model Combinations

| S.No | Data-Model Combination                               | Accuracy | F1    |
|------|------------------------------------------------------|----------|-------|
| 1    | **A1**: SCD → LSTM                                   | 0.8872   | 0.8870  |
| 2    | **A2**: RVD → CNN                                    | 0.8709   | 0.8707 |
| 3    | **A2**: RVD → ViViT                                  | 0.9431   | 0.93   |
| 4    | **A3**: OFD → CNN                                    | 0.9271   | 0.9269 |
| 5    | **A3**: OFD → ViViT                                  | 0.9340   | 0.94   |
| 6    | **A4**: SCD → LSTM, RVD → CNN                        | 0.9434   | 0.9431 |
| 7    | **A5**: SCD → LSTM, OFD → CNN                        | 0.9487   | 0.9484 |
| 8    | **A6**: RVD → CNN, OFD → CNN                         | 0.9210   | 0.9220 |
| 9    | **A6**: RVD → ViViT, OFD → CNN                       | 0.9502   | 0.9511 |
| 10   | **A6**: RVD → CNN, OFD → ViViT                       | 0.9189   | 0.9185 |
| 11   | **A7**: RVD → ViViT, OFD → CNN, SCD → LSTM           | **0.9531**| **0.9523** |
| 12   | **A7**: RVD → CNN, OFD → ViViT, SCD → LSTM           | 0.9364   | 0.9362 |

**Table 3:** ASD Classification of Various Data-Model Combinations

From **Tables 2 & 3**, we observed higher accuracy values for framework **RVD → ViViT, OFD → CNN, SCD → LSTM**, so we have used this framework to do the both action classification and ASD classification tasks simultaneoulsy in same framework, **Figure 2** displays detailed connects of Multimodal Transformer Framework , and performace evalution of the model is over 95% for both tasks. This show cases the strength of multimodal framework along with the privacy preserving dataset strength for giving subtle body movement information. 

<div align="center">
  <img src="MMASD_FINAL_COPU.jpg" alt="Flowchart of Data Processing (don't forget to change ROMP image)" width="80%">

  <p><strong>Figure 2:</strong> Multimodal Transformer Framework</p>
</div>


# Code and Data

The complete code for each data-model combination is available in the code folder, and a sample dataset is provided in the data folder.

The data folder is organized into three subfolders:

1. ROMP: Contains 3D body mesh coordinates stored in .npy files.
2. 3D-Skeleton Coordinates: Includes 3D joint coordinates in CSV format. Each row represents a frame, and each column corresponds to a joint.
3. Optical Flow: Contains dense optical flow data.



