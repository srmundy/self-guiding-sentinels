# Self-Guiding Sentinels
A novel computer vision system for detecting physical attacks on surveillance equipment while distinguishing between legitimate maintenance and malicious tampering attempts.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Threat Level System](#threat-level-system)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Citations](#citations)
- [License](#license)

## 🔍 Overview

Self-Guiding Sentinels is an advanced surveillance system designed to protect security cameras from physical tampering. Unlike traditional systems that focus on network security, our solution specializes in detecting physical attacks while accurately distinguishing them from legitimate maintenance activities or environmental factors.

The system employs a hybrid architecture combining Vision Transformers with lightweight diffusion components, achieving 93.3% accuracy in tampering detection. When attacks are detected, it implements a graduated threat response system with anthropomorphic audio cues to deter attackers.

## ✨ Features

- **High Accuracy Detection:** 93.3% accuracy in distinguishing between tampering and maintenance
- **Low False Positives:** Only 3.2% false positive rate compared to 14-20% in traditional systems
- **Real-time Performance:** (WIP)
- **Graduated Threat Levels:** Five threat levels with appropriate responses
- **Anthropomorphic Deterrence:** Uses simulated "pain" expressions to deter attacks (WIP)
- **Pre-attack Detection:** Identifies suspicious behavior 3-5 seconds before tampering

## 🏗️ Architecture

The system architecture consists of three main components:

1. **Vision Transformer Backbone**
   - 16×16 patch embedding
   - 12 transformer layers
   - 768 embedding dimension
   - 12 attention heads
   - 86M parameters

2. **Lightweight Diffusion Component**
   - Trajectory prediction
   - Uncertainty modeling
   - Occlusion filling
   - 12M parameters

3. **Classification Head**
   - 4-class detection (Normal, Environmental, Maintenance, Attack)
   - 0.4M parameters

![Architecture Diagram](docs/images/architecture.png)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sarahmundy/self-guiding-sentinels.git
   cd self-guiding-sentinels
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Data Preparation

The system works with the UHCTD dataset and can be supplemented with synthetic data.

1. Download the UHCTD dataset from [http://qil.uh.edu/main/datasets/](http://qil.uh.edu/main/datasets/)

2. Organize your dataset in the following structure:
   ```
   data/
   ├── videos/
   │   ├── video_001.mp4
   │   ├── video_002.mp4
   │   └── ...
   ├── annotations.csv
   ```

3. The annotations CSV file should include the following columns:
   - `video_name`: Name of the video file
   - `event_type`: Class label ("normal", "environmental", "maintenance", "attack")
   - `start_frame`: Starting frame of the event
   - `end_frame`: Ending frame of the event

4. Generate synthetic data (optional):
   ```bash
   python scripts/generate_synthetic_data.py --output-dir data/synthetic --num-samples 1000
   ```

### Training

To train the Self-Guiding Sentinels model (note that you need to add the data to the data directory):

```bash
python train.py \
  --data-root data/videos \
  --annotation-file data/annotations.csv \
  --output-dir output/model \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --epochs 100 \
  --mixed-precision \
  --clip-length 16 \
  --frame-interval 5
```

Key parameters:

- `--data-root`: Directory containing video files
- `--annotation-file`: Path to annotations CSV
- `--output-dir`: Directory to save model checkpoints and logs
- `--batch-size`: Training batch size
- `--learning-rate`: Initial learning rate
- `--epochs`: Number of training epochs
- `--mixed-precision`: Enable mixed precision training (faster with compatible GPUs)
- `--clip-length`: Number of frames in each video clip
- `--frame-interval`: Interval between sampled frames

For a complete list of options, run `python train.py --help`.

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py \
  --model-path output/model/best_model.pth \
  --data-root data/videos \
  --annotation-file data/test_annotations.csv \
  --output-dir output/evaluation
```

Key parameters:

- `--model-path`: Path to the saved model checkpoint
- `--data-root`: Directory containing test video files
- `--annotation-file`: Path to test annotations CSV
- `--output-dir`: Directory to save evaluation results

### Inference

For real-time inference on video streams:

```bash
python inference.py \
  --model-path output/model/best_model.pth \
  --input-source 0  # Use camera index or video file path
  --threat-level-system
  --audio-response
```

Key parameters:

- `--model-path`: Path to the saved model checkpoint
- `--input-source`: Camera index or video file path
- `--threat-level-system`: Enable graduated threat level responses
- `--audio-response`: Enable anthropomorphic audio responses (WIP)

## 📊 Results

The Self-Guiding Sentinels system achieves:

| Metric | Value |
|--------|-------|
| Accuracy | 94.3% |
| F1 Score | 0.89 |
| ROC AUC | 0.857 |
| False Positive Rate | 3.2% |
| False Negative Rate | 5.7% |
| Inference Time | 42ms/frame |

Performance comparison:

| Model | Accuracy | F1 Score | ROC AUC | FPR | FNR |
|-------|----------|----------|---------|-----|-----|
| UHCTD Baseline | 73.5% | 0.71 | 0.723 | 14.2% | 12.3% |
| ViT-only | 87.2% | 0.82 | 0.812 | 7.5% | 9.8% |
| Commercial System A | 80.4% | 0.76 | 0.784 | 12.8% | 10.2% |
| Commercial System B | 83.7% | 0.79 | 0.803 | 9.5% | 11.4% |
| Human Experts | 94.0% | 0.91 | 0.932 | 4.5% | 3.2% |
| **Self-Guiding Sentinels** | **94.3%** | **0.89** | **0.857** | **3.2%** | **5.7%** |

## 🚨 Threat Level System

The system implements five graduated threat levels:

### Level 1: Low Concern (Monitoring)
- **Triggers**: Initial detection (50-65% confidence)
- **Response**: Increased sampling rate, logging, alert neighboring cameras

### Level 2: Moderate Concern (Warning)
- **Triggers**: Medium confidence detection (65-80%)
- **Response**: Soft audio warning ("Camera monitoring active"), increased resolution

### Level 3: High Concern (Deterrence)
- **Triggers**: High confidence detection (80-90%)
- **Response**: Moderate volume alert ("Please step away from the camera"), security notification

### Level 4: Critical Threat (Active Intervention)
- **Triggers**: Very high confidence (>90%)
- **Response**: Loud anthropomorphic expression ("Camera under attack! Pain detected!"), lighting activation

### Level 5: Breach (Emergency Protocol)
- **Triggers**: Camera offline after attack detection
- **Response**: Emergency protocols, neighboring cameras increase monitoring

## 📁 Project Structure

```
self-guiding-sentinels/
├── data/                  # Data directory
├── docs/                  # Documentation
├── model/                 # Model architecture
│   ├── __init__.py
│   ├── model.py           # Main model implementation
│   └── components/        # Model components
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   └── visualize.py       # Visualization utilities
├── scripts/               # Helper scripts
│   ├── generate_synthetic_data.py
│   └── process_dataset.py
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Real-time inference script
├── requirements.txt       # Dependencies
├── LICENSE
└── README.md
```

## 📊 Datasets

### UHCTD Dataset
- **Source**: [http://qil.uh.edu/main/datasets/](http://qil.uh.edu/main/datasets/)
- **Size**: 10,000 videos


### Synthetic Dataset
- Generated using diffusion models
- 15,000 synthetic attack scenarios
- Includes variations in lighting, viewing angles, and attack methods
- **Classes**: Normal operation, Environmental factors, Legitimate maintenance, Physical attacks


## 📝 Citations

If you use Self-Guiding Sentinels in your research, please cite:

```bibtex
@article{mundy2025self,
  title={Self-Guiding Sentinels: An Accurate Physical Attack Surveillance System},
  author={Mundy, Sarah},
  journal={arXiv preprint TBD},
  year={2025}
}
```




---

For questions or collaboration opportunities, please contact: Sarah.Mundy@columbia.edu
