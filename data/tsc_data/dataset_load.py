import torch
import torch.utils.data as data

from .augmentations import DataTransform

    
class Load_Dataset(data.Dataset):
    def __init__(self, dataset, target, config=None):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        self.target = target

    def __getitem__(self, index):
      
        return self.dataset[index], self.target[index]
    
    def __len__(self):
        return len(self.target)
