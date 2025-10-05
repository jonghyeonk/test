# 📄 Project Title
**Event-Driven Neuromorphic Gaze Decoding via e-Skin Electrooculography**

---

## 🔹 Overview
This repository provides the code for a **two-stage neuromorphic gaze decoding pipeline**:

- **LSTM-based sequence learner** – analyzes temporal EOG signals and highlights where attention should be focused.  
- **SNN-based neuromorphic decoder** – receives attention-guided spiking input and performs event-driven gaze decoding.  

---

## 🔹 Repository Structure
```plaintext
.
├── LSTM/
│   ├── ckpts/             # pretrained LSTM weights
│   ├── train.py           # Train the LSTM model on raw Jeonju dataset
│   ├── inference.py       # Run inference, generate attention-guided segments
│   ├── evaluate.py        # Evaluate LSTM performance
│   ├── model.py           # LSTM model architecture
│
├── data_jeonju_0904 /
│   ├── task1_v2_x4        # Original dataset for training a four-section classification problem
│   └── task2_v2_x8        # Original dataset for training an eight-section classification problem
│
├── figures_rawdata /
│   ├── figure3            # raw dataset for figure 3
│   └── figure4            # raw dataset for figure 4
│
├── SNN/
│   ├── preprocess.py      # Convert raw EOG → spiking data using LSTM attention
│   ├── train.py           # Train the SNN model with spike-based representation
│   ├── test.py            # Evaluate SNN decoding performance
│   └── checkpoints/       # Trained SNN weights
```
0. Requirements.txt (python version == 3.11)
```plaintext
numpy==1.23.5
pandas==2.1.4
scipy==1.9.3
matplotlib==3.7.5
seaborn==0.13.2
torch==2.3.0+cu121
snntorch==0.9.1
scikit-learn==1.4.2
torchvision==0.18.0+cu121
```

1. LSTM Training & Inference
```plaintext
python LSTM/train.py
python LSTM/inference.py
```

2. Attention-Guided Preprocessing
```plaintext
python LSTM/train.py
python LSTM/inference.py
```

3. SNN Training & Evaluation
```plaintext
python SNN/train.py
python SNN/test.py
python SNN/evaluate.py
```
