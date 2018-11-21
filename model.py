import torch.nn as nn
from torch.nn import init
from torchvision.models.vgg import vgg16_bn
import numpy as np



class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            )
        self.weight_init()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def weight_init(self):
        for layer in self.features:
            self._layer_init(layer)
        for layer in self.classifier:
            self._layer_init(layer)


    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
        # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            init.xavier_normal(m.weight)

