import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def get_openimages6_dataloaders(dataset_root, transforms, train_batch_size, num_workers=12):
    train_transforms, val_transforms = transforms
    train_dataset = OpenImages6Dataset(dataset_root, *train_transforms)
    val_dataset = OpenImages6Dataset(dataset_root, *val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class OpenImages6Dataset(Dataset):
    def __init__(
        self,
        dataset_root,
        img_transforms,
        text_transforms,
        images_ids='/home/tolik/fiftyone/open-images-v6/train/metadata/image_ids.csv'
    ):
        self.dataset_root = dataset_root

        images_ids_path = os.path.join(dataset_root, 'train/metadata/image_ids.csv')
        self.images_ids = pd.read_csv(images_ids, nrows=1000).values

        self.img_transforms = img_transforms
        self.text_transforms = text_transforms

    def __getitem__(self, i):
        values = self.images_ids[i]
        img = os.path.join(self.dataset_root, f"train/data/{values[0]}.jpg")
        img = Image.open(img)
        text = values[7]

        img = self.img_transforms(to_rgb(img))
        text = self.text_transforms(text)

        return img, text

    def __len__(self):
        return len(self.images_ids)


def to_rgb(img):
    rgbimg = Image.new('RGB', img.size)
    rgbimg.paste(img)

    return rgbimg

def collate_fn(data):
    xs = {'image': [], 'text': []}

    for img, text in data:
        img = img.unsqueeze(0)
        text = text.unsqueeze(0)
        xs['image'].append(img)
        xs['text'].append(text)

    xs['image'] = torch.cat(xs['image'], 0)
    xs['text'] = torch.cat(xs['text'], 0)

    return xs, xs
