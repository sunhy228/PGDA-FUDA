import sys
import torch
import argparse
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10, ImageFolder
from torch.utils.data.distributed import DistributedSampler
from utils import datautils

import numpy as np
from typing import Tuple

def load_cifar10(batchsize:int, numworkers:int, tr:bool) -> Tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root='./test/ProtoDiff/cifar10',
                        train=tr,
                        download=True,
                        transform=trans
                    )
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        drop_last=True
                    )
    return trainloader

def load_cifar10_target(batchsize:int, numworkers:int) -> Tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root='./test/ProtoDiff/cifar10',
                        train=False,
                        download=True,
                        transform=trans
                    )
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        drop_last=True
                    )
    return trainloader

def load_cifar10_source(batchsize:int, numworkers:int) -> Tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root='./test/ProtoDiff/cifar10',
                        train=True,
                        download=True,
                        transform=trans
                    )
    trainloader_sl = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        drop_last=True
                    )
    trainloader_su = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        drop_last=True
                    )
    return trainloader_sl, trainloader_su

def load_cifar10_sample(batchsize:int, numworkers:int) -> Tuple[DataLoader, DistributedSampler]:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10(
                        root='./test/ProtoDiff/cifar10',
                        train=False,
                        download=True,
                        transform=trans,
                    )
    # sort
    labels = data_train.targets[:160]
    _, indices = torch.sort(torch.tensor(labels))
    train_images = []
    train_labels = []
    for idx in indices:
        train_images.append(data_train[idx][0])
        train_labels.append(data_train[idx][1])

    trainloader = DataLoader(
                        #data_train,
                        torch.utils.data.TensorDataset(torch.stack(train_images, dim=0),torch.tensor(train_labels)),
                        batch_size=batchsize,
                        num_workers=numworkers,
                        drop_last=True,
                        shuffle=False
                    )
    return trainloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5


def load_stl10(batchsize:int, numworkers:int, image_size: int) -> Tuple[DataLoader, DistributedSampler]:
    if image_size == -1:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size,image_size)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        
    data_train = STL10(
                        root='./datasets/stl10',
                        split='train',
                        download=True,
                        transform=trans
                        )
    data_test = STL10(
                        root='./',
                        split='test',
                        download=True,
                        transform=trans
                        )
    # merge train and test
    data_train.data = np.concatenate([data_train.data,data_test.data],axis=0)
    data_train.labels = np.concatenate([data_train.labels,data_test.labels],axis=0)

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        sampler=sampler,
                        drop_last=True
                    )
    return trainloader,sampler

def load_sample(params:argparse.Namespace) -> Tuple[DataLoader, DistributedSampler]:
    name = params.name  # office
    num_workers = params.numworkers # 0
    domain = params.dataset # {'source': 'amazon', 'target': 'dslr'}'

    image_size = params.shape # 224
    aug_name = 'aug_0' # aug_0
    raw = "raw"

    num_class = datautils.get_class_num(f'../data/splits/{name}/{domain}.txt') # 31
    class_map = datautils.get_class_map(f'../data/splits/{name}/{domain}.txt') # 类名映射

    batch_size = params.genbatch # 64
            # Sampling datasets
    train_dataset = datautils.create_dataset(
        name,
        domain,
        suffix="",
        ret_index=True,
        image_transform=aug_name,
        use_mean_std=False,
        image_size=image_size,
    )
    # [index,feature,label] train of source/target
    sample_loader = datautils.create_loader(
        train_dataset,
        batch_size,
        is_train=False,
        num_workers=num_workers,
    )
    '''
    for batch in train_loader:
        print(batch[0].shape)  # [64] #index
        print(batch[1].shape)  # [64, 3, 224, 224] #feature
        print(batch[2].shape)  # [64] #label
    '''

    return sample_loader


def load_tiny_imagenet(batchsize:int, numworkers:int, image_size: int, image_dir: str) -> Tuple[DataLoader, DistributedSampler]:
    if image_size == -1:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size,image_size)),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

    data_train = ImageFolder(image_dir, transform=trans)
    # train data path, *./datasets/tiny_imagenet/train

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size=batchsize,
                        num_workers=numworkers,
                        sampler=sampler,
                        drop_last=True
                    )
    return trainloader,sampler

def get_dataset(source, target, batchsize, numworkers):
    if source == 'cifar10':
        loader_sl, loader_su = load_cifar10_source(batchsize, numworkers)
        class_number = 10
    else:
        raise NotImplementedError
    if target == 'cifar10':
        loader_tl = load_cifar10_target(batchsize, numworkers)
        class_number = 10
    else:
        raise NotImplementedError
    
    return loader_sl, loader_su, loader_tl
    
