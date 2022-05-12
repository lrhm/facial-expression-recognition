import os
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from .utils.data_manager import DataManger


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.data_location = params.data_location
        emotions = os.listdir(params.data_location)
        folders = tuple(
            (em, os.path.join(params.data_location, em)) for em in emotions
        )
        all_images = []
        for emotion, folder in folders:
            images = os.listdir(folder)
            for image in images:
                all_images.append((emotion, os.path.join(folder, image)))
        ipdb.set_trace()


    def train_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(self.data_location, train=True,)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )

    def val_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(self.data_location, train=False,)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )

    def test_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(self.data_location, train=False,)
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=3,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        data_location: str = "/mnt/facial_expression_dataset/",
        train: bool = True,
    ):
        self.data_location = data_location
        self.train = train
        emotions = os.listdir(data_location)
        folders = tuple(os.path.join(data_location, em) for em in emotions)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, : self.in_seq_len]
        y = self.data[idx, self.out_seq_len :]
        return x, y


def test():
    data_module = CustomDataModule()
    train_dataloader = data_module.train_dataloader()
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)
    """
    train_dl, test_dl = get_loaders(
        "/mnt/tmp/multi_channel_train_test",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
        in_seq_len=8,
        out_seq_len=4,
    )
    for i, (x, y) in enumerate(tqdm(train_dl)):
        # plt.imshow(x[0, 0, 0].cpu())
        # plt.show()
        # print(x.shape)
        # return
        # print(f"iteration: {i}")
        pass
    """
    # reads file in h5 format


if __name__ == "__main__":
    test()


