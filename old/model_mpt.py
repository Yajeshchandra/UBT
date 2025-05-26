import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import os
from PIL import Image
import torch.nn.functional as F
from .dataset import BiometricDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import time

class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim=256, dropout_p=0.3):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.res1 = nn.Conv2d(1, 64, kernel_size=1) if 1 != 64 else nn.Identity()
        self.res2 = nn.Conv2d(64, 128, kernel_size=1) if 64 != 128 else nn.Identity()
        self.res3 = nn.Conv2d(128, 256, kernel_size=1) if 128 != 256 else nn.Identity()
        self.res4 = nn.Conv2d(256, 512, kernel_size=1) if 256 != 512 else nn.Identity()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        # Depthwise convolution on feature maps before flattening
        self.depthwise_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512)
        self.depthwise_bn = nn.BatchNorm2d(512)
        
        self.fc = nn.Linear(512 * 4 * 4, embedding_dim)  # Adjust based on input size
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(embedding_dim)

        
    def forward(self, x):

        identity = self.res1(x)
        x = self.bn1(self.conv1(x))
        x = F.relu(x + identity)
        x = self.pool(x)

        identity = self.res2(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        x = self.pool(x)

        identity = self.res3(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity)
        x = self.pool(x)

        identity = self.res4(x)
        x = self.bn4(self.conv4(x))
        x = F.relu(x + identity)
        
        # Apply depthwise convolution directly on feature maps
        # x = self.depthwise_conv(x)
        # x = self.depthwise_bn(x)
        # x = F.relu(x)
        
        # Pooling and FC
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # [B, 512*4*4]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.ln(x)  # Normalize embedding
        return x

class FusionTransformer(nn.Module):
    def __init__(self, embedding_dim=256, nhead=8, num_layers=4, dropout_p=0.1, num_modalities=3, num_prompts=3):
        super(FusionTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # Transformer encoder for global context
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                       nhead=nhead, 
                                       dim_feedforward=512, 
                                       dropout=dropout_p, 
                                       batch_first=True),
            num_layers=num_layers
        )

        # Positional encodings
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_modalities + num_prompts, embedding_dim), requires_grad=True)
        
        # Layer norms for input and output
        self.ln_in = nn.LayerNorm(embedding_dim)
        self.ln_out = nn.LayerNorm(embedding_dim)
        
        # MPT: Learnable prompt embeddings
        self.num_prompts = num_prompts
        self.prompt_emb = nn.Parameter(torch.randn(num_prompts, embedding_dim))
        self.prompt_conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.prompt_relu = nn.ReLU()
        
        # Final projection
        self.fc = nn.Linear(embedding_dim * (num_modalities + num_prompts), embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize the prompt embeddings and positional encoder
        nn.init.xavier_uniform_(self.prompt_emb)
        nn.init.xavier_uniform_(self.pos_encoder)
        
    def forward(self, *embeddings):
        batch_size = embeddings[0].size(0)
        
        # Stack embeddings for transformer input
        modality_embeddings = torch.stack(embeddings, dim=1)  # [batch_size, num_modalities, embedding_dim]
        
        # Prepare prompt embeddings
        prompt = self.prompt_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_prompts, embedding_dim]
        prompt = prompt.permute(0, 2, 1)  # [batch_size, embedding_dim, num_prompts]
        prompt = self.prompt_conv(prompt)  # [batch_size, embedding_dim, num_prompts]
        prompt = self.prompt_relu(prompt)
        prompt = prompt.permute(0, 2, 1)  # [batch_size, num_prompts, embedding_dim]
        
        # Concatenate modality embeddings and prompts
        x = torch.cat([modality_embeddings, prompt], dim=1)  # [batch_size, num_modalities + num_prompts, embedding_dim]
        
        # Apply layer norm and add positional encodings
        x = self.ln_in(x) + self.pos_encoder
        
        # Apply transformer for global context
        x = self.transformer(x)
        
        # Reshape and project to final embedding
        x = x.reshape(batch_size, -1)  # [batch_size, (num_modalities + num_prompts) * embedding_dim]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.ln_out(x)
        return x



# Complete Model
class BiometricModel(nn.Module):
    def __init__(self, embedding_dim=256, num_modalities=3):
        super(BiometricModel, self).__init__()

        self.num_modalities = num_modalities

        self.periocular_cnn = CNNEmbedding(embedding_dim)
        self.forehead_cnn = CNNEmbedding(embedding_dim)
        self.iris_cnn = CNNEmbedding(embedding_dim)

        self.fusion_transformer = FusionTransformer(embedding_dim=256, num_modalities=num_modalities, num_prompts=3)
        
    def forward(self, periocular, forehead, iris):
        periocular_emb = self.periocular_cnn(periocular)
        forehead_emb = self.forehead_cnn(forehead)
        iris_emb = self.iris_cnn(iris)
        emb = [periocular_emb, forehead_emb, iris_emb]
        fused_emb = self.fusion_transformer(*emb)
        fused_emb = F.normalize(fused_emb, dim=1)
        return fused_emb