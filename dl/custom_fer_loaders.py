import ipdb as ipdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as ttf, utils


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
    def __init__(self, data, transforms, fancy_transform=True):
        self.data = data

        self.affine_transform = ttf.RandomAffine(degrees=3)
        self.random_crop = ttf.RandomCrop(48, padding=4, padding_mode="reflect")
        self.random_horizontal_flip = ttf.RandomHorizontalFlip()
        self.random_rotation = ttf.RandomRotation(20)

        self.fancy_transform = fancy_transform

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, target = self.data[i]
        if self.transforms is not None:
            # random transformations based on probability
            if self.fancy_transform:
                prob = torch.rand(1)[0]
                motecarlo = torch.rand(1)[0]
                # if prob > 0.7:
                #     image = self.affine_transform(image)

                if prob < 0.5:
                    image = self.random_crop(image)
                    if motecarlo > 0.3:
                        image = self.random_rotation(image)

                if prob < 0.7:
                    image = self.random_horizontal_flip(image)

            else:
                image = self.transforms(image)
            # image = self.transforms(image)
        # return a tuple of data transformed data and corresponding label
        return (image, target)
