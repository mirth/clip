import numpy as np
from torchvision import transforms
import imgaug.augmenters as iaa


def aug1():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
    ])

    return transforms.Compose([
        transforms.RandomCrop(64),
        np.asarray,
        transforms.Lambda(lambda x: seq(image=x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
