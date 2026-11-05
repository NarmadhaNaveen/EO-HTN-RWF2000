# EO-HTN: Edge-Optimized Hybrid Transformer Networks

### Context-Aware Adaptation for Threat Classification in Visual Security Systems

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

## 1. Project Overview

EO-HTN (Edge-Optimized Hybrid Transformer Networks) is a deep-learning research project designed for visual threat classification in security surveillance videos.

The system combines convolutional neural networks (CNNs), Transformer-based temporal feature learning, and a context-aware adaptation mechanism to analyze video frames and classify activities into two categories:

- **Fight**
- **NonFight**

The objective is to develop a compact and efficient video-classification architecture that can learn spatial and temporal information while exploring calibration and edge-optimization considerations.

The project currently uses the RWF-2000 dataset and includes a training pipeline, evaluation scripts, threshold analysis, and an interactive Streamlit dashboard.

## 2. Problem Statement

Traditional video-surveillance systems often require continuous human monitoring. Detecting potentially violent activities from large amounts of video footage can be challenging.

This project investigates an automated deep-learning approach that analyzes video frames and classifies fight-related activities.

The proposed EO-HTN architecture combines spatial feature extraction and temporal relationships to improve video understanding.

## 3. Proposed Architecture

The model consists of the following components:

1. **CNN Feature Extraction:** Extracts spatial features from individual video frames.
2. **Frame Embedding:** Converts extracted features into 128-dimensional representations.
3. **Positional Encoding:** Adds temporal position information to frame embeddings.
4. **Transformer Encoder:** Learns relationships between video frames.
5. **Context-Aware Gate:** Uses global temporal context to adaptively weight frame features.
6. **Classification Head:** Predicts Fight or NonFight.

### Architecture Flow

```text
Input Surveillance Video
          |
          v
   Frame Extraction
          |
          v
    CNN Feature Stem
          |
          v
  Frame Feature Embeddings
          |
          v
  Positional Embeddings
          |
          v
 Transformer Encoder
          |
          v
  Context-Aware Gate
          |
          v
   Classification Head
          |
          v
   Fight / NonFight
```

## 4. Dataset

**Dataset:** RWF-2000 (Real-World Fighting)

The current experiment uses the following dataset split:

| Split | Fight | NonFight | Total |
|---|---:|---:|---:|
| Training | 800 | 800 | 1,600 |
| Validation | 200 | 200 | 400 |
| Total | 1,000 | 1,000 | 2,000 |

The dataset contains real-world surveillance videos.

The raw dataset is not included in this GitHub repository because of its large storage requirements.

## 5. Model Configuration

| Component | Configuration |
|---|---|
| Input frames | 8 |
| Input resolution | 96 × 96 |
| CNN channels | 32, 64, 96 |
| Embedding dimension | 128 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Context-aware adaptation | Enabled |
| Output classes | 2 |
| Trainable parameters | 411,426 |

## 6. Technologies Used

- Python 3.12.7
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib

## 7. Training and Improvement

The project includes an improved V2 training pipeline.

V2 introduces:

- Random temporal frame sampling during training.
- Centre-based frame sampling during validation.
- Training-only horizontal flipping.
- Mild brightness and contrast augmentation.
- Consistent normalization.
- AdamW optimizer.
- ReduceLROnPlateau learning-rate scheduling.
- Early stopping.
- Best-checkpoint selection using validation F1-score.

The improved training experiment completed 14 epochs before early stopping.

## 8. Experimental Results

The following are the measured results from the selected V2 checkpoint on the 400-video validation split.

| Metric | V2 Validation Result |
|---|---:|
| Accuracy | 71.25% |
| Precision | 63.84% |
| Recall | 98.00% |
| F1-score | 77.32% |
| Brier Score | 0.3781 |
| Expected Calibration Error | 0.0601 |

### Confusion Matrix

Class order: NonFight, Fight.

| Actual / Predicted | NonFight | Fight |
|---|---:|---:|
| NonFight | 184 | 16 |
| Fight | 4 | 196 |

The selected checkpoint was obtained at epoch 9 based on validation F1
