import torchvision.models as models
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),

                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(True)
            )

        self.class_classifier = nn.Sequential(
                nn.Linear(64*64*64, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout2d(),

                nn.Linear(1024, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(True),

                nn.Linear(100, 2),
            )

    def forward(self, x):
        x = self.features(x).view(-1, 64*64*64)
        #print(x.size())
        output = self.class_classifier(x)
        return output
