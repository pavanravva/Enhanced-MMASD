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
Before extracting the provacy features, the raw video recording undergo some pre-processing steps, the individual persons are extracted form the video, and then each person privacy features are extracted. The detailes steps of features extraction can be illustrated in Figure 1:
![Flowchart of Data Processing (don't forget to change ROMP image)](data_flow_chart_mmasd.jpg)

**Figure 1:** Flowchart of Data Processing







