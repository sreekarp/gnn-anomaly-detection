import torch
import torch.optim as optim
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.preprocessing import StandardScaler
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

# Scale features
scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
data = data.to(device)
print(" Data loaded and preprocessed.")

# Initialize Model and Optimizer
in_channels = data.num_node_features
hidden_channels_encoder = 128
out_channels_encoder = 64
hidden_channels_decoder = 128

# Instantiate the components of our new semi-supervised model
encoder = GATEncoder(in_channels, hidden_channels_encoder, out_channels_encoder)
attr_decoder = AttributeDecoder(out_channels_encoder, hidden_channels_decoder, in_channels)
model = SemiSupervisedGAE(encoder, attr_decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)

print(" Model and optimizer initialized.")

# Training Loop
def train():
    model.train()
    optimizer.zero_grad()
    
    z = model.encode(data.x, data.edge_index)
    
    # Calculate the new combined loss, passing the labels and train_mask
    loss = model.loss(data.x, data.y, z, data.edge_index, data.train_mask, alpha=0.2, beta=0.2)
    
    loss.backward()
    optimizer.step()
    
    return float(loss)

print("\nStarting model training...")
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}')

print(" Training complete.")

# Save the Trained Model
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

model_path = 'saved_models/semi_supervised_gae.pt'
torch.save(model.state_dict(), model_path)

print(f"\n Model saved to {model_path}")
