import ipdb as ipdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils


classes = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# transform data added horizontal flip and random cropping

# dataset class
class DataFER(Dataset):

    def __init__(self, images, labels, transforms):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # convert pixels from string to list of int
        data = [int(el) for el in self.X[i].split(" ")]
        # reshape into 48 x 48 x 1 np array for transform
        data = np.asarray(data).astype(np.uint8).reshape(48, 48, 1)
        data = self.transforms(data)
        label = self.y[i]
        # return a tuple of data transformed data and corresponding label
        return (data, label)
