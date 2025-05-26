import torch
import torch.nn as nn
import torch.nn.functional as F


# InfoNCE Loss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)  # Exclude self-similarity
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = exp_sim.sum(dim=1)
        
        loss = -torch.log(pos_sum / (neg_sum + 1e-8) + 1e-8)

        pos_sim = (sim_matrix * pos_mask).sum() / pos_mask.sum()
        neg_sim = (sim_matrix * (1 - pos_mask)).sum() / (1 - pos_mask + 1e-8).sum()
        # print(f"Pos Sim: {pos_sim.item():.4f}, Neg Sim: {neg_sim.item():.4f}")

        return loss.mean()


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute pairwise distances
        batch_size = embeddings.size(0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        y = (labels == labels.T).float()
        y.fill_diagonal_(0)  # Exclude self-pairs
        
        # Create mask for negative pairs (different class)
        neg_mask = 1 - y
        neg_mask.fill_diagonal_(0)  # Exclude self-pairs
        
        # Compute contrastive loss
        positive_loss = y * torch.pow(dist_matrix, 2)
        negative_loss = neg_mask * torch.pow(torch.clamp(self.margin - dist_matrix, min=0.0), 2)
        
        # Sum over all pairs and normalize
        loss = (positive_loss + negative_loss).sum() / (y.sum() + neg_mask.sum() + 1e-8)
        
        # Optional: Calculate average positive and negative distances for monitoring
        pos_dist = (dist_matrix * y).sum() / (y.sum() + 1e-8)
        neg_dist = (dist_matrix * neg_mask).sum() / (neg_mask.sum() + 1e-8)
        # print(f"Pos Dist: {pos_dist.item():.4f}, Neg Dist: {neg_dist.item():.4f}")
        
        return loss

