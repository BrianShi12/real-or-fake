import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class RealFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None, root_dir='data/'):
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        img_path = os.path.join(self.root_dir, row['file_name'])  # join root_dir
        img = Image.open(img_path).convert("RGB")
        
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
