import numpy
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from conf import INPUT_SHAPE, NORMALIZE_STD, INPUT_SIZE, ROTATE_ANGLE, NORMALIZE_MEAN
import torchvision
import os

# data loader and transform
loader = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

transformers = {
    "train": transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.RandomRotation(ROTATE_ANGLE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ]),
    "test": transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ]),
    "val": transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
}


def image_show(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = numpy.array(NORMALIZE_MEAN)
    std = numpy.array(NORMALIZE_STD)
    inp = std * inp + mean
    inp = numpy.clip(inp, 0, 1)
    # disable axes
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.imshow(inp)
    plt.show()


def image_loader(image_name):
    image = PIL.Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image


class DatasetLoader:
    def __init__(self, batch_size, workers, data_path):
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.data = None
        self.sizes = None

    def load(self):
        dset = {
            'train': ImageFolder(os.path.join(self.data_path, 'train'), transform=transformers['train']),
            'val': ImageFolder(os.path.join(self.data_path, 'val'), transform=transformers['val']),
            'test': ImageFolder(os.path.join(self.data_path, 'test'), transform=transformers['test']),
        }

        self.data = {
            'train': DataLoader(dset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.workers),
            'val': DataLoader(dset['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.workers),
            'test': DataLoader(dset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.workers),
        }
        self.sizes = {
            'train': len(dset['train']),
            'val': len(dset['val']),
            'test': len(dset['test']),
        }

        return self

    def next(self, ds):
        return next(iter(self.data[ds]))


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = numpy.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model, self.path)
        self.val_loss_min = val_loss
