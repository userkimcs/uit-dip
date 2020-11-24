import logging
import os
import sys

import torch.nn as nn
import torch.optim as optim
import torch
import time
import copy
from tqdm import tqdm
# do not remove
from torchvision.models import *
from helpers import DatasetLoader
from helpers import EarlyStopping

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ModelTrainer(nn.Module):
    def __init__(self, dataset: DatasetLoader, model_name='vgg16', num_epochs=1, **es):
        super(ModelTrainer, self).__init__()
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.model_name = model_name
        # create model from imgnet
        self.model = globals()[model_name](pretrained=True)

        # freeze
        for params in self.model.parameters():
            params.requires_grad = False

        # add last layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2),
            nn.LogSoftmax(dim=1)
        )

        self.CUDA = torch.cuda.is_available()

        # es
        self.es = EarlyStopping(patience=es.get('patience', 5), delta=es.get('delta', 0), path=f"models/{self.model_name}.pt")

    def forward(self, x):
        return self.model(x)

    def fit(self):

        optimizer = optim.Adam(self.model.fc.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.NLLLoss()
        since = time.time()

        if self.CUDA:
            self.model = self.model.cuda()

        for epoch in tqdm(range(0, self.num_epochs)):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
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
                            scheduler.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset.sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.sizes[phase]

                if phase == 'test':
                    logger.info("Early stop validating...")
                    self.es(epoch_loss, self.model)
                    if self.es.early_stop:
                        logger.warning("Early stopping")
                        break

                time_elapsed = time.time() - since
                logger.info(f"{phase} loss:  {epoch_loss}  acc: {epoch_acc}")
                logger.info(f'Time = {time_elapsed // 60}m {time_elapsed % 600}s')

                with open("logs.txt", 'a') as wf:
                    wf.write(f"{phase}\n")
                    wf.write(f"Epoch: {epoch}\n")
                    wf.write(f"Model = {self.model_name}\n")
                    wf.write(f"Time = {time_elapsed // 60}m {time_elapsed % 600}s\n")
                    wf.write(f"Loss = {epoch_loss}\n")
                    wf.write(f"Acc = {epoch_acc}\n")
                    wf.write(f"=========================\n")

        torch.save(self.model.state_dict(), f"models/{self.model_name}_{self.num_epochs}.pt")
        return self.model
