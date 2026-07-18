import os
from pathlib import Path
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

CURRENT_DIR = Path.cwd()
DATA_DIR = CURRENT_DIR / "Fish"
IMG_SIZE = 224 # Resize target dim
BATCH_SIZE = 32
VALIDATION_SIZE = 0.15 # Percent of data for validation
TEST_SIZE = 0.15 # Percent of data for testing (Remaining data after validation and testing is for training [0.3])
RANDOM_SEED = 42
NUM_WORKERS = 2

# Get classes of fish and the count of each class
def index_dataset(data_dir: Path):
    class_names = sorted([dir.name for dir in data_dir.iterdir() if dir.is_dir()])
    class_to_index = {name: index for index, name in enumerate(class_names)}

    filepaths, labels = [], []
    for class_name in class_names:
        class_dir = data_dir / class_name
        for img_path in class_dir.glob("*.jpg"):
            filepaths.append(str(img_path))
            labels.append(class_to_index[class_name])

    if not filepaths:
        raise RuntimeError(
            f"No .jpg files found in {data_dir}. Check DATA_DIR path and image files"
        )

    print("Class distribution:")
    counts = Counter(labels)
    for name, index in class_to_index.items():
        print(f"  {name:10s}: {counts.get(index, 0)}")

    return filepaths, labels, class_to_index

def stratified_split(filepaths, labels, VALIDATION_SIZE=VALIDATION_SIZE, test_size=TEST_SIZE, seed=RANDOM_SEED):
    trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
        filepaths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    validation_fraction_of_remainder = VALIDATION_SIZE / (1.0 - test_size)

    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        trainval_paths,
        trainval_labels,
        test_size=validation_fraction_of_remainder,
        stratify=trainval_labels,
        random_state=seed,
    )

    print(f"\nSplit sizes -> train: {len(train_paths)}, val: {len(validation_paths)}, "
          f"test: {len(test_paths)}")

    return (train_paths, train_labels), (validation_paths, validation_labels), (test_paths, test_labels)

class FishDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

eval_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def build_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    filepaths, labels, class_to_index = index_dataset(data_dir)
    (train_paths, train_labels), (validation_paths, validation_labels), (test_paths, test_labels) = stratified_split(filepaths, labels)

    train_dataset = FishDataset(train_paths, train_labels, transform=train_transform)
    validation_dataset = FishDataset(validation_paths, validation_labels, transform=eval_transform)
    test_dataset = FishDataset(test_paths, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader, class_to_index

if __name__ == "__main__":
    train_loader, validation_loader, test_loader, class_to_index = build_dataloaders()

    print(f"\nclass_to_index: {class_to_index}")

    images, labels = next(iter(train_loader))
    print(f"Sample train batch -> images: {images.shape}, labels: {labels.shape}")