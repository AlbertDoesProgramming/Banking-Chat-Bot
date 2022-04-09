import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
        
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples
    
    