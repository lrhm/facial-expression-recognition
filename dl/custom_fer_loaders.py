import ipdb as ipdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils


classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# transform data added horizontal flip and random cropping

# dataset class
class DataFER(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, target = self.data[i]
        if self.transforms is not None:
            image = self.transforms(image)
        # return a tuple of data transformed data and corresponding label
        return (image, target)
