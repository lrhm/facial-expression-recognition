import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image


from torchvision.datasets.utils import verify_str_arg, check_integrity
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms

import ipdb


class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _RESOURCES = {
        "fer": ("train.csv", "f8428a1edbd21e88f42c73edd2a14f95"),
    }

    def __init__(
        self,
        data_location: str = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(data_location, transform=transform, target_transform=target_transform)

        base_folder = data_location
        file_name, md5 = self._RESOURCES["fer"]
        data_file = data_location #+ file_name

        
        trsfm = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.transform = trsfm

        if not check_integrity(str(data_file), md5=md5):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        with open(data_file, "r", newline="") as file:
            self._samples = [
                (
                    torch.tensor(
                        [int(idx) for idx in row["pixels"].split()], dtype=torch.uint8
                    ).reshape(48, 48),
                    int(row["emotion"]) if "emotion" in row else None,
                )
                for row in csv.DictReader(file)
            ]

        test_percentage = 0.2
        if train:
            self._split = "train"
            self._samples = self._samples[: -int(len(self._samples) * test_percentage)]
        else:
            self._split = "test"
            self._samples = self._samples[-int(len(self._samples) * test_percentage) :]

        # ipdb.set_trace()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # one hot encoding
        target = torch.eye(7)[target]

        return image, target

    def extra_repr(self) -> str:
        return f"split={self._split}"
