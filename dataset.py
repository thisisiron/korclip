import os
import json
from typing import Callable, Optional

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        tokenizer = None
    ):
        self.root = root
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def _load_image(self, idx: int):
        image_path = self.df.file_path[idx]
        image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        return image

    def _load_text(self, idx):
        return self.df.caption_ko[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        text = self._load_text(index)

        if self.transform is not None:
            image = self.transform(image)

        return image, text 

    def __len__(self) -> int:
        return len(self.df)
