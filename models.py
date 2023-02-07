# https://github.com/alevine0/smoothingSplittingNoise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
from torchvision import models as base_models
from datasets import *
import math 


class Forecaster(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        self.norm = NormalizeLayer(get_normalization_shape(dataset), device,
                                   **get_normalization_stats(dataset))

    def forward(self, x):
        raise NotImplementedError

    def forecast(self, theta):
        return Categorical(logits=theta)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class ResNet(Forecaster):

    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = nn.DataParallel(base_models.resnet50(num_classes=1000))
        self.norm = nn.DataParallel(self.norm)
        self.norm = self.norm.to(device)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)


class WideResNet(Forecaster):
    
    def __init__(self, dataset, device):
        super().__init__(dataset, device)
        self.model = nn.DataParallel(WideResNetBase(depth=40, widen_factor=2,
                                                    num_classes=get_num_labels(dataset)))
        self.norm = nn.DataParallel(self.norm)
        self.norm = self.norm.to(device)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)




class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Examples:
        (64, 3, 32, 32) [CIFAR] => normalizes across channels
        (64, 8) [UCI]  => normalizes across features
    """
    def __init__(self, dim, device, mu=None, sigma=None):
        super().__init__()
        self.dim = dim
        if mu and sigma:
            self.mu = nn.Parameter(torch.tensor(mu, device=device).reshape(dim), requires_grad=False)
            self.log_sig = nn.Parameter(torch.log(torch.tensor(sigma, device=device)).reshape(dim), requires_grad=False)
            self.initialized = True
        else:
            raise ValueError

    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
            self.initialized = True
        return (x - self.mu) / torch.exp(self.log_sig)

    def initialize_parameters(self, x):
        with torch.no_grad():
            mu = x.view(x.shape[0], x.shape[1], -1).mean((0, 2))
            std = x.view(x.shape[0], x.shape[1], -1).std((0, 2))
            self.mu.copy_(mu.data.view(self.dim))
            self.log_sig.copy_(torch.log(std).data.view(self.dim))



class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and \
                            nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(i == 0 and in_planes or \
                                     out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetBase(nn.Module):

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):

        super(WideResNetBase, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
