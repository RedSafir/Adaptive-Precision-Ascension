import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APALinear, APAConfig

def _create_linear(in_features, out_features, bias=True, config=None, use_apa=True):
    if use_apa:
        return APALinear(in_features, out_features, bias=bias, config=config)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

# VGG-16 configuration: 13 conv layers + 5 max pools
VGG16_CONFIG = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]

class VGG16(nn.Module):
    """VGG-16 Architecture adapted for CIFAR-10 (32x32 images).

    Features:
        - 13 Convolutional layers with Batch Normalization and ReLU
        - 5 Max Pooling layers
    Classifier:
        - 3 Fully Connected layers using APALinear (or standard nn.Linear if use_apa=False)
        - Dropout (p=0.5) for regularization
    """
    def __init__(self, num_classes=10, config=None, use_apa=True, dropout=0.5):
        super().__init__()
        self.use_apa = use_apa
        self.config = config
        
        # 1. Feature Extractor (Convolutional Backbone)
        self.features = self._make_layers(VGG16_CONFIG)
        
        # 2. Classifier (Dense / Fully Connected Layers)
        # For 32x32 input, 5 MaxPool layers reduce spatial dimensions: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Feature map before classifier is 512 x 1 x 1 = 512
        self.fc1 = _create_linear(512, 512, bias=True, config=config, use_apa=use_apa)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = _create_linear(512, 512, bias=True, config=config, use_apa=use_apa)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = _create_linear(512, num_classes, bias=True, config=config, use_apa=use_apa)
        
        self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        out = self.features(x)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Classifier with APA
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out
