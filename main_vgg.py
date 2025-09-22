import torchvision.models as models
import torch
import torch.nn as nn
from Feature_Correction_Module import *
from HeightWidthChannelEulerProcessor import *
from VTFNet import *


class FM(nn.Module):
    def __init__(self, num_classes=100):
        super(FM, self).__init__()
        vgg11 = models.vgg16(pretrained=True)
        vgg11.eval()
        self.feature_extractor_x = vgg11.features
        self.feature_extractor_y = vgg11.features

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