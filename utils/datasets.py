import torch
import pandas as pd
from PIL import Image
from typing import Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

class CXRDatasets(Dataset):
    def __init__(self, dataset_name: str, imagedir_filepath: Path, labels_filepath: Path,
                 cxr_labels: Tuple[str, ...], transforms: transforms.Compose):
        self.imagedir_fpath = imagedir_filepath
        self.labels_fpath = labels_filepath
        self.df = pd.read_csv(labels_filepath)
        self.cxr_labels = cxr_labels
        self.transforms = transforms
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int)-> Dict[str, Any]:
        row = self.df.iloc[idx]
        if self.dataset_name == 'nihcxr':
            img = Image.open(
                self.imagedir_fpath / f"{row['Image Index     ']}"
            ).convert('L')

        elif self.dataset_name == 'vincxr':
            img = Image.open(
                self.imagedir_fpath / f"{row['image_id']}.png"
            ).convert('L')

        elif self.dataset_name == 'chexpert':
            img = Image.open(
                self.imagedir_fpath / f"{row['Path'][:-3]}png"
            ).convert('L')

        else:
            raise NotImplementedError(f'{self.dataset_name} is not supported')

        img = img.convert('RGB')
        # transforms
        if self.transforms is not None:
            img = self.transforms(img)
        return {
            'img': img, 
            'labels': torch.tensor(row[self.cxr_labels].tolist()),
            # 'names': [col for col in self.cxr_labels if row[col] == 1]
            }

class DictImageFolder(ImageFolder):
    def __getitem__(self, index): # type: ignore
        image, label = super().__getitem__(index)
        return {'img': image, 'labels': label}