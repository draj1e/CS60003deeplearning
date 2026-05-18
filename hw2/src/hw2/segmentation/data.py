from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF


class StanfordBackgroundDataset(Dataset):
    def __init__(self, root: str | Path, image_size: tuple[int, int]) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.images = sorted((self.root / "images").glob("*.jpg")) + sorted((self.root / "images").glob("*.png"))
        self.labels_dir = self.root / "labels"
        self.labels_raw_dir = self.root / "labels_raw"
        if not self.images:
            raise FileNotFoundError(f"未找到图像，请按 images/ 与 labels/ 组织 Stanford Background 数据集: {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        label_path = self.labels_dir / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB")
        if label_path.exists():
            label_array = np.array(Image.open(label_path), dtype=np.int64)
        else:
            raw_label_path = self.labels_raw_dir / f"{image_path.stem}.regions.txt"
            label_array = np.loadtxt(raw_label_path, dtype=np.int64)
        image = TF.resize(image, self.image_size, antialias=True)
        label = Image.fromarray(label_array.astype(np.uint8), mode="L")
        label = TF.resize(label, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        label_tensor = torch.as_tensor(np.array(label), dtype=torch.long)
        return image_tensor, label_tensor


def build_segmentation_loaders(data_root: str | Path, image_size: list[int], batch_size: int, num_workers: int, seed: int) -> tuple[DataLoader, DataLoader]:
    dataset = StanfordBackgroundDataset(data_root, tuple(image_size))
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [train_size, val_size], generator=generator)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs)
