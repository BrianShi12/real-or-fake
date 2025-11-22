import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class RealFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        img_path = row['image_path'] # gets the image path from the dataframe
        img = Image.open(img_path).convert("RGB")
        
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
