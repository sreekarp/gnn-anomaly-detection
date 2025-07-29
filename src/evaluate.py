import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import os

# Import the updated model class
from models import GATEncoder, AttributeDecoder, SemiSupervisedGAE

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and Preprocess Data
try:
    dataset = EllipticBitcoinDataset(root='./data/EllipticBitcoinDataset')
    data = dataset[0]
except Exception as e:
    print(f" Error loading dataset: {e}")
    exit()

scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
data = data.to(device)
print(" Data loaded and preprocessed.")

#  Prepare Test Data
# test_mask provided by the dataset for a fair evaluation
test_mask = data.test_mask
test_indices = torch.where(test_mask)[0]

# 4. Load the Trained Model
in_channels = data.num_node_features
hidden_channels_encoder = 128
out_channels_encoder = 64
hidden_channels_decoder = 128
model_path = 'saved_models/semi_supervised_gae.pt'

encoder = GATEncoder(in_channels, hidden_channels_encoder, out_channels_encoder)
attr_decoder = AttributeDecoder(out_channels_encoder, hidden_channels_decoder, in_channels)
model = SemiSupervisedGAE(encoder, attr_decoder).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f" Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f" Error: Model file not found at {model_path}. Please run train.py first.")
    exit()

model.eval()

# 5. Calculate Anomaly Scores (Corrected Logic)
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)
    
    # The classifier was explicitly trained to separate illicit from licit nodes.
    # A higher logit value indicates a higher probability of being an anomaly.
    logits = model.classify(z)
    anomaly_scores_tensor = logits.squeeze() # Remove unnecessary dimensions

# Evaluate GNN Performance
gnn_anomaly_scores = anomaly_scores_tensor[test_indices].cpu().numpy()
true_labels = data.y[test_indices].cpu().numpy()

if len(np.unique(true_labels)) < 2:
    print(" Error: The test set contains only one class. Cannot calculate AUC-ROC score.")
else:
    gnn_auc_score = roc_auc_score(true_labels, gnn_anomaly_scores)
    print("\n--- GNN Model Performance ---")
    print(f"Semi-Supervised GAE AUC-ROC Score: {gnn_auc_score:.4f}")

