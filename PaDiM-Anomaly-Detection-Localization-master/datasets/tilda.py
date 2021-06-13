import os
import os.path as osp

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision import datasets

root_path = 'tilda/ncd1/c2/r2/images'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

run_cases = [
#'tilda/ncd1/c1/r1/images', 
#'tilda/ncd1/c1/r3/images', 
#'tilda/ncd1/c2/r2/images', 
#'tilda/ncd1/c2/r3/images', 
'tilda/ncd2/c3/r1/images', 
'tilda/ncd2/c3/r3/images', 
'tilda/ncd2/c4/r1/images',  
'tilda/ncd2/c4/r3/images'
]


def get_dataset(path, is_train=True,
                resize=256, cropsize=224):
    transform = T.Compose([
        T.Resize(resize, Image.ANTIALIAS),
        T.CenterCrop(cropsize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    train_dset = datasets.ImageFolder(osp.join(path, 'train'), transform=transform)
    test_dset = datasets.ImageFolder(osp.join(path,'test'), transform=transform)
    return train_dset, test_dset
