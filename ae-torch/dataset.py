import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os.path as osp

split = 'carpet'
folder = 'mvtec/%s' % split


def get_trainval_loader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(256, padding=16),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.22, 0.22, 0.22])
    ])
    dset = datasets.ImageFolder(osp.join(folder, 'train'), transform=transform_train)
    dset_size = len(dset)
    train_size = int(0.7 * dset_size)
    val_size = dset_size - train_size
    trainset, valset = random_split(dset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(43))
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)
    return trainloader, valloader


def get_test_loader():
    transform_test = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.22, 0.22, 0.22])
    ])
    testset = datasets.ImageFolder(osp.join(folder, 'test'), transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False,
                            num_workers=2)
    return testloader
