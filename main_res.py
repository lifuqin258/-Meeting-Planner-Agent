import torchvision.models as models
import torch
import torch.nn as nn
from Feature_Correction_Module import *
from HeightWidthChannelEulerProcessor import *
from VTFNet import *


class FM(nn.Module):
    def __init__(self, num_classes=100):
        super(FM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.eval()
        modules = list(resnet18.children())[:-2]  # 只保留到 layer4
        self.feature_extractor_x = nn.Sequential(*modules)
        self.feature_extractor_y = nn.Sequential(*modules)

        self.HeightWidthChannelEulerProcessor = HeightWidthChannelEulerProcessor(input_dimension=512)
        self.feature_extractor = FCM(dim=512)

        self.VTFNet = VTFNet()

        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        x, y = self.feature_extractor(x, y)

        fusion = self.VTFNet(x, y)

        fusion = self.HeightWidthChannelEulerProcessor(fusion)

        fusion = self.dropout(fusion)
        fusion = fusion.reshape(fusion.size(0), -1)
        fusion = self.fc(fusion)
        return fusion