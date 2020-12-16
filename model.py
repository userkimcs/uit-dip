import logging
import os
import sys
import time
import re
import scipy.ndimage as ndimg
from datetime import datetime
from matplotlib import pyplot as plt
import PIL
import numpy
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
# do not remove
from torchvision.models import *
from torchvision.transforms import transforms
from tqdm import tqdm


from helpers import DatasetLoader, image_loader
from helpers import EarlyStopping

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

MODEL_FOLDER = os.path.join(os.path.curdir, "models")


class ModelTrainer(nn.Module):
    def __init__(self, dataset: DatasetLoader, model_name='resnet18', num_epochs=1, **es):
        super(ModelTrainer, self).__init__()
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.model_name = model_name
        # create model from imgnet
        self.model = globals()[model_name](pretrained=True)

        # freeze
        for params in self.model.parameters():
            params.requires_grad = False

        if self.model_name == 'resnet18':
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Linear(512, 2),
                nn.LogSoftmax(dim=1)
            )
        elif self.model_name == 'densenet121':
            self.model.classifier = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 2),
                nn.LogSoftmax(dim=1)
            )
        elif self.model_name == 'mnasnet0_5':
            self.model.classifier[1] = nn.Sequential(
                nn.BatchNorm1d(1280),
                nn.Linear(1280, 2),
                nn.LogSoftmax(dim=1)
            )
        elif self.model_name == 'inception_v3':
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Linear(2048, 2),
                nn.LogSoftmax(dim=1)
            )
        elif self.model_name == 'mobilenet_v2':
            self.model.classifier[1] = nn.Sequential(
                nn.BatchNorm1d(1280),
                nn.Linear(1280, 2),
                nn.LogSoftmax(dim=1)
            )
        elif self.model_name == 'vgg11':
            self.model.classifier[-1] = nn.Sequential(
                nn.BatchNorm1d(4096),
                nn.Linear(4096, 2),
                nn.LogSoftmax(dim=1)
            )
        else:
            sys.exit(1)

        self.CUDA = torch.cuda.is_available()

        # es
        self.es = EarlyStopping(patience=es.get('patience', 5), delta=es.get('delta', 0),
                                path=f"models/{self.model_name}.pt")

    def forward(self, x):
        return self.model(x)

    def run(self):
        model_file = f"{MODEL_FOLDER}/{self.model_name}_{self.num_epochs}.pt"
        if os.path.isfile(model_file):
            print(f'Model `f"{model_file}.pt"` trained.')
            return

        optimizer = optim.Adam(self.model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.NLLLoss()
        since = time.time()

        if self.CUDA:
            self.model = self.model.cuda()

        for epoch in tqdm(range(0, self.num_epochs)):
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                    if epoch > 0:
                        scheduler.step()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in tqdm(self.dataset.data[phase]):
                    if self.CUDA:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset.sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.sizes[phase]

                if phase == 'test':
                    logger.info("Early stop validating...")
                    self.es(epoch_loss, self.model)
                    if self.es.early_stop:
                        logger.warning("Early stopping")
                        return

                time_elapsed = time.time() - since
                logger.info(f"{phase} loss:  {epoch_loss}  acc: {epoch_acc}")
                logger.info(f'Time = {time_elapsed // 60}m {time_elapsed % 600}s')

                with open(f"logs_{self.model_name}.txt", 'a') as wf:
                    wf.write(f"{phase}\n")
                    wf.write(f"Epoch: {epoch}\n")
                    wf.write(f"Model = {self.model_name}\n")
                    wf.write(f"Time = {time_elapsed // 60}m {time_elapsed % 600}s\n")
                    wf.write(f"Loss = {epoch_loss}\n")
                    wf.write(f"Acc = {epoch_acc}\n")
                    wf.write(f"=========================\n\n")

        if not os.path.isdir(MODEL_FOLDER):
            os.mkdir(MODEL_FOLDER)

        torch.save(self.model, f"{model_file}.pt")
        return self.model


class LayerActivations:
    features = []

    def __init__(self, model):
        self.hooks = []
        # model.layer4 is the last layer of our network before the Global Average Pooling layer
        # (last convolutional layer).
        self.hooks.append(model.layer4.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


class ModelEval(nn.Module):
    def __init__(self, dataset, model_path):
        super(ModelEval, self).__init__()
        self.dataset = dataset
        self.model_path = model_path
        self.CUDA = torch.cuda.is_available()
        device = 'cuda' if self.CUDA else 'cpu'

        self.model = torch.load(model_path, map_location=torch.device(device))
        self.model.eval()

    def run(self):
        criterion = nn.NLLLoss()
        running_loss = running_corrects = 0

        for inputs, labels in tqdm(self.dataset.data['test']):
            if self.CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        logger.info(f"Model {self.model_path}")
        logger.info(f"Loss {running_loss / self.dataset.sizes['test']}")
        logger.info(f"Acc {running_corrects / self.dataset.sizes['test']}")


class ImageVisualize:
    def __init__(self, model_path):
        self.CUDA = torch.cuda.is_available()
        device = 'cuda' if self.CUDA  else 'cpu'
        self.model = torch.load(model_path, map_location=torch.device(device))
        self.model.eval()
        self.acts = LayerActivations(self.model)

    def run(self, img_path):
        logger.info("Load layer act")

        img = image_loader(img_path)
        if self.CUDA:
            self.model = self.model.cuda()

        logps = self.model(img)
        ps = torch.exp(logps)
        out_features = self.acts.features[0]
        out_features = torch.squeeze(out_features, dim=0)
        out_features = numpy.transpose(out_features.detach().numpy(), axes=(1, 2, 0))
        pred = numpy.argmax(ps.detach())
        w = self.model.fc[1].weight[pred, :]  # ignore batch normalize layer
        cam = numpy.dot(out_features, w.detach())
        class_activation = ndimg.zoom(cam, zoom=(32, 32), order=1)
        img = numpy.squeeze(img, axis=0)
        img = numpy.transpose(img, (1, 2, 0))

        plt.imshow(img, cmap='Greys', alpha=1)
        plt.imshow(class_activation, cmap='jet', alpha=0.4)
        plt.show()

