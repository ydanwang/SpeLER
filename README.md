# SpeLER

A PyTorch implementation for Bridging Class Imbalance and Partial Labeling via Spectral-Balanced Energy Propagation for Skeleton-based Action Recognition

## Overview

Skeleton-based action recognition faces class imbalance and insufficient labeling problems in real-world applications. Existing methods typically address these issues separately, lacking a unified framework that can effectively handle both issues simultaneously while considering their inherent relationships. Our theoretical analysis reveals two fundamental connections between these problems. First, class imbalance systematically shifts the eigenvalue spectrum of normalized affinity matrices, compromising both convergence and accuracy of label propagation. Second, boundary samples are critical for model training under imbalanced conditions but are often mistakenly excluded by conventional reliability metrics, which focus on relative class differences rather than holistic connectivity patterns. Built upon these theoretical findings, we propose SpeLER (**Spe**ctral-balanced **L**abel Propagation with **E**ergy-based Tightened **R**eliability), which introduces a spectral balancing technique that explicitly counteracts spectral shifts by incorporating class distribution information. Meanwhile, a propagation energy-based tightened reliability measure is proposed to better preserve crucial boundary samples by evaluating holistic connectivity patterns. Extensive experiments on six public datasets demonstrate that SpeLER consistently outperforms state-of-the-art methods, validating both our theoretical findings and practical effectiveness.

## Eigenvalue distribution under class imbalance
![image](https://github.com/user-attachments/assets/031e2cf3-af08-4b95-82c1-7c5c2af0e009)

## Requirements

- PyTorch 1.8+

```bash
pip install -r requirements.txt
```

## Project Structure

```
SpeLER/
├── config_files/         # Configuration files for different datasets
├── data/                 # Data loading and preprocessing (SAR and Classical TSC)
├── model/               # Model architectures and components
├── utils/               # Utility functions and helpers
├── results/             # Training results and logs
└── main.py           
```

## Dataset

### Classical Time-series classification

**Raw data**

- **Epilepsy** contains single-channel EEG measurements from 500 subjects. For each subject, the brain activity was recorded for 23.6 seconds. The dataset was then divided and shuffled (to mitigate sample-subject association) into 11,500 samples of 1 second each, sampled at 178 Hz. The raw dataset features 5 different classification labels corresponding to different status of the subject or location of measurement - eyes open, eyes closed, EEG measured in healthy brain region, EEG measured where the tumor was located, and, finally, the subject experiencing seizure episode. To emphasize the distinction between positive and negative samples in terms of epilepsy, We merge the first 4 classes into one and each time series sample has a binary label describing if the associated subject is experiencing seizure or not. There are 11,500 EEG samples in total. To evaluate the performance of pre-trained model on small fine-tuning dataset, we choose a tiny set (60 samples; 30 samples for each class) for fine-tuning and assess the model with a validation set (20 samples; 10 sample for each class). The model with best validation performance is use to make prediction on test set (the remaining 11,420 samples). The [raw dataset](https://repositori.upf.edu/handle/10230/42894) is distributed under the Creative Commons License (CC-BY) 4.0.

-  **HAR** contains recordings of 30 health volunteers performing six daily activities such as walking, walking upstairs, walking downstairs, sitting, standing, and laying. The prediction labels are the six activities. The wearable sensors on a smartphone measure triaxial linear acceleration and triaxial angular velocity at 50 Hz. After preprocessing and isolating out gravitational acceleration from body acceleration, there are nine channels in total. To line up the semantic domain with the channels in the dataset use during fine-tuning **Gesture** we only use the three channels of body linear accelerations. The [raw dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.
- **SleepEDF** provides single-channel EEG signals (100Hz), covering five sleep stages: Wake (W), Non-REM (N1, N2, N3), and REM.

GPU: 4090

### Skeleton-based action recognition

**Raw Data**

1. Download the [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) dataset:
   
    i. nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)
    
    ii. nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)
    
    iii. Extract above files to ./data/nturgbd_raw
   
2. Download the Kinetics-Skeleton dataset
   
## Model Training

### Classical Time-series classification

```bash
python main_TSC.py --dataset HAR --configs HAR --labeled_ratio *
```

```bash
python main_TSC.py --dataset Epilepsy --configs Epilepsy --labeled_ratio *
```

### Skeleton-based action recognition

```bash
python main.py --dataset NTU --configs NTU60 \
    --labeled_ratio * --imb_ratio_l * 
```

2. For **Kinetics-Skeleton** dataset:
```bash
python main.py --dataset K400 --configs K400 \
    --labeled_ratio * --imb_ratio_l *
```

### Important Arguments

- `--dataset`: Choose from ['NTU',  'K400']
- `--configs`: Configuration file name
- `--labeled_ratio`: Percentage of labeled data
- `--imb_ratio_l`: Imbalance ratio for labeled data
- `--imb_ratio_u`: Imbalance ratio for unlabeled data
- `--batch_size`: Training batch size
- `--epochs`: Total training epochs
- `--epochs_unsupervised`: Number of unsupervised training epochs
- `--device`: Device to use (cuda or cpu)

GPU: L4

## Model Architecture

The model consists of several key components:

1. Time Branch
   - Graph Convolutional Network
   - Multi-scale Temporal Convolution
   - Transformer Encoder (Optional)

2. Frequency Branch
   - Frequency-aware Gated Convolution
   - Cross-domain Feature Fusion

3. Label Propagation
   - Spectral balance
   - Propagation energy assessment
