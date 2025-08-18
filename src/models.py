import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GAE

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class AttributeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AttributeDecoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, z):
        z = F.relu(self.lin1(z))
        z = self.lin2(z)
        return z

class SemiSupervisedGAE(GAE):
    def __init__(self, encoder, attr_decoder, num_classes=2):
        super(SemiSupervisedGAE, self).__init__(encoder)
        self.attr_decoder = attr_decoder
        # A simple linear layer to act as the classifier
        self.classifier = nn.Linear(encoder.conv2.out_channels, 1) # Output for binary classification

    def decode_attr(self, z):
        return self.attr_decoder(z)
    
    def classify(self, z):
        return self.classifier(z)

    def loss(self, x, y, z, edge_index, train_mask, alpha=0.5, beta=0.3):
        """
        Calculates a combined loss:
        1. Structural Loss (unsupervised)
        2. Attribute Loss (unsupervised)
        3. Classification Loss (supervised, only on labeled nodes)
        """
        # Unsupervised losses (calculated on all nodes)
        structural_loss = self.recon_loss(z, edge_index)
        x_reconstructed = self.decode_attr(z)
        attribute_loss = F.mse_loss(x_reconstructed, x)
        
        # Supervised loss (calculated only on labeled training nodes)
        classification_loss = 0
        if train_mask.sum() > 0:
            # Use binary_cross_entropy_with_logits for numerical stability
            classification_loss = F.binary_cross_entropy_with_logits(
                self.classify(z).squeeze()[train_mask], 
                y[train_mask].float()
            )

        # Combine the losses with weights
        return (alpha * structural_loss + 
                beta * attribute_loss + 
                (1 - alpha - beta) * classification_loss)
