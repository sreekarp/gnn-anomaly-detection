# Advanced Anomaly Detection using Graph Neural Networks

This repository contains the complete code for a project focused on detecting illicit transactions in the Elliptic Bitcoin dataset. It demonstrates an iterative approach to building a robust anomaly detection system, starting from simple unsupervised models and culminating in a high-performing semi-supervised Graph Neural Network (GNN).

The final model successfully leverages a GAT-based architecture to achieve an **AUC-ROC score of approximately 0.85**, showcasing its strong capability to distinguish between licit and illicit financial activities.

---

## Project Structure

The repository is organized to clearly separate the final, working source code from the experimental scripts that document the project's journey.

- **gnn-anomaly-detection/**
  - **data/**
    - (This directory is created automatically by the scripts to store the dataset)
  - **notebooks/**
    - attempt_1_latent_norm.py  
    - attempt_2_recon_error.py  
    - data_Set_and_iso_forest.py
  - **saved_models/**
    - semi_supervised_gae.pt
  - **src/**
    - __init__.py  
    - models.py  
    - train.py  
    - evaluate.py
  - .gitignore  
  - requirements.txt  
  - README.md

---

## Getting Started

Follow these steps to set up the environment and run the project on your local machine.

### Prerequisites

* Python 3.8+
* NVIDIA GPU with CUDA support (strongly recommended for training performance)
* Conda package manager

### Installation Guide

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/gnn-anomaly-detection.git](https://github.com/your-username/gnn-anomaly-detection.git)
    cd gnn-anomaly-detection
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    conda create --name gnn-env python=3.9
    conda activate gnn-env
    ```

3.  **Install PyTorch with GPU Support:**
    This is the most important step. Go to the [PyTorch official website](https://pytorch.org/get-started/locally/) and use the interactive tool to find the correct installation command for your specific system (OS, package manager, CUDA version). For example, a common command is:
    ```bash
    pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Install Remaining Dependencies:**
    Once PyTorch is correctly installed, use the `requirements.txt` file to install the other necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
---

## How to Run the Project

The final, best-performing model is contained within the `src/` directory.

### 1. Train the Model

Execute the `train.py` script to start the training process. The script will automatically download the dataset, preprocess it, train the semi-supervised GNN for 200 epochs, and save the final model weights to the `saved_models/` directory.

```bash
python src/train.py
```

### 2.  Evaluate the Model

After the training is complete, run the evaluate.py script. This will load the saved model and compute its performance on the held-out test set, printing the final AUC-ROC score.

```bash
python src/train.py
```
You should see an output score of approximately 0.84-0.85.
