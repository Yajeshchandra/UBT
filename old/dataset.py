from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class BiometricDataset(Dataset):
    def __init__(self, root_dir, transform=None, instances_per_person=4, split='train'):
        """
        Args:
            root_dir (str): Root directory containing 'iris', 'periocular', and 'forehead' folders.
            transform (callable, optional): Optional transform to be applied to images.
            instances_per_person (int): Number of instances to return per person (default: 2).
            split (str): 'train' or 'test' to select the dataset split.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.instances_per_person = instances_per_person
        self.split = split  # 'train' or 'test'
        
        # Get list of person IDs
        self.iris_dir = os.path.join(root_dir, f'iris/{split}')
        self.periocular_dir = os.path.join(root_dir, f'periocular/{split}')
        self.forehead_dir = os.path.join(root_dir, f'forehead/{split}')
        self.person_ids = [f for f in sorted(os.listdir(self.periocular_dir)) if f != '.DS_Store']
        
        # Store image paths for each person
        self.iris_images = {}
        self.periocular_images = {}
        self.forehead_images = {}
        
        for person_id in self.person_ids:
            iris_person_dir = os.path.join(self.iris_dir, person_id)
            periocular_person_dir = os.path.join(self.periocular_dir, person_id)
            forehead_person_dir = os.path.join(self.forehead_dir, person_id)
            
            self.iris_images[person_id] = [
                os.path.join(iris_person_dir, img)
                for img in sorted(os.listdir(iris_person_dir))
                if img.endswith('.jpg')
            ]
            self.periocular_images[person_id] = [
                os.path.join(periocular_person_dir, img)
                for img in sorted(os.listdir(periocular_person_dir))
                if img.endswith('.jpg')
            ]            
            self.forehead_images[person_id] = [
                os.path.join(forehead_person_dir, img)
                for img in sorted(os.listdir(forehead_person_dir))
                if img.endswith('.jpg')
            ]
        
        # Assign numerical labels
        self.label_map = {pid: idx for idx, pid in enumerate(self.person_ids)}
        self.num_persons = len(self.person_ids)
    

    def __len__(self):
        return self.num_persons
    

    def __getitem__(self, idx):
        """
        Returns 2 instances for a person, each with a randomly selected iris and fingerprint.
        """
        person_id = self.person_ids[idx]
        label = self.label_map[person_id]
        
        perioculars = []
        irises = []
        foreheads = []
        labels = []
        
        # Generate 2 instances
        for _ in range(self.instances_per_person):
            # Randomly select images
            iris_img_path = random.choice(self.iris_images[person_id])
            periocular_image_path = random.choice(self.periocular_images[person_id])
            forehead_image_path = random.choice(self.forehead_images[person_id])
            
            # Load images
            iris_img = Image.open(iris_img_path).convert('L')
            periocular_img = Image.open(periocular_image_path).convert('L')
            forehead_img = Image.open(forehead_image_path).convert('L')
            
            # Apply transforms
            if self.transform is not None:
                iris_img = self.transform(iris_img)
                periocular_img = self.transform(periocular_img)
                forehead_img = self.transform(forehead_img)
            
            irises.append(iris_img)
            perioculars.append(periocular_img)
            foreheads.append(forehead_img)
            labels.append(label)
        
        return {
            'irises': torch.stack(irises),  # Shape: [2, 1, 128, 128]
            'perioculars': torch.stack(perioculars),  # Shape: [2, 1, 128, 128]
            'foreheads': torch.stack(foreheads),  # Shape: [2, 1, 128, 128]
            'labels': torch.tensor(labels, dtype=torch.long)  # Shape: [2]
        }

