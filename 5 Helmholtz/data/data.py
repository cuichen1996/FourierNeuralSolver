import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


class KappaDataGenerator:
    def __init__(self, hight: int, width: int) -> None:
        self.hight = hight
        self.width = width

    def load_data(self, batch_size: int = 1, data="stl10"):
        if data == "stl10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale(1),
                    transforms.Resize((self.hight, self.width), antialias=True),
                    transforms.GaussianBlur(5),
                    MinMaxScalerVectorized(feature_range=(0.25, 1)),
                ]
            )

            trainset = torchvision.datasets.STL10(
                root="./data", split="test", download=True, transform=transform
            )

        elif data == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.GaussianBlur(5, 3),
                    transforms.Grayscale(1),
                    transforms.Resize((self.hight, self.width), antialias=True),
                    MinMaxScalerVectorized(feature_range=(0.25, 1)),
                ]
            )
            trainset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        self.trainset = trainset

    def generate_kappa(self) -> torch.Tensor:
        image, _ = next(iter(self.trainloader))
        return image


class MinMaxScalerVectorized:
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor
            A tensor with scaled features using requested preprocessor.
        """
        # Feature range
        a, b = self.feature_range
        return (tensor - tensor.min()) * (b - a) / (tensor.max() - tensor.min()) + a


class OpenFWIDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        samples_per_file: int,
        transform=None,
        target_transform=None,
        preload=True,
    ) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload
        self.samples_per_file = samples_per_file
        self.samples = []
        self.prefix_data = "model"

        if self.preload:
            file_names = sorted(
                [
                    name
                    for name in os.listdir(self.folder_path)
                    if os.path.isfile(os.path.join(self.folder_path, name))
                    and name.startswith(self.prefix_data)
                ]
            )
            self.samples = [
                np.load(f"{self.folder_path}/{file_name}") for file_name in file_names
            ]

    def __getitem__(self, index) -> torch.Tensor:
        batch_idx, sample_idx = (
            index // self.samples_per_file,
            index % self.samples_per_file,
        )
        if self.preload:
            data = self.samples[batch_idx][sample_idx]
        else:
            data = np.load(f"{self.folder_path}/{self.prefix_data}{index}.npy")
            data = data[sample_idx]

        if self.transform:
            data = data.transpose(1, 2, 0)
            data = self.transform(data)

        return data, data

    def __len__(self) -> int:
        return self.samples_per_file * self._get_num_of_batches()

    def _get_num_of_batches(self):
        return len(
            [
                name
                for name in os.listdir(self.folder_path)
                if os.path.isfile(os.path.join(self.folder_path, name))
                and name.startswith(self.prefix_data)
            ]
        )
