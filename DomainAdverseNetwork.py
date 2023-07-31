import numpy as np
import torch
import torch.nn as nn

from Config import Config

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            # (N,1,32,32)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # (N,64,32,32)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N,64,16,16)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (N,128,16,16)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N,128,8,8)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (N,256,8,8)
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N,256,4,4)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # (N,512,4,4)
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N,512,2,2)

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),  # (N,1024,2,2)
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N,1024,1,1)

            nn.Flatten() # (N,1024)
        )

    def forward(self, x):
        """
        :param x: (N,1,32,32)
        :return:
        """
        return self.net(x)


class CategoryClassifier(nn.Module):
    def __init__(self):
        super(CategoryClassifier, self).__init__()
        self.net = nn.Sequential(
            # (N,1024)
            nn.Linear(1024, 512), # (N,512)
            nn.ReLU(),
            nn.Linear(512, 256), # (N,256)
            nn.ReLU(),
            nn.Linear(256, 10) # (N,10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.net = nn.Sequential(
            # (N,1024)
            nn.Linear(1024, 512), # (N,512)
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),

            nn.Linear(512, 512),  # (N,512)
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),

            nn.Linear(512, 512),  # (N,512)
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),

            nn.Linear(512, 1) # (N,1)
        )

    def forward(self, x):
        return self.net(x)


class UpperClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(UpperClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.category_classifier = CategoryClassifier()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_classifier(x)
        return x


class LowerDiscriminator(nn.Module):
    def __init__(self, feature_extractor):
        super(LowerDiscriminator, self).__init__()
        self.feature_extractor = feature_extractor
        self.domain_classifier = DomainClassifier()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.domain_classifier(x)
        return x


class DomainAdverseNetwork(nn.Module):
    def __init__(self):
        super(DomainAdverseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_classifier = CategoryClassifier()
        self.domain_classifier = DomainClassifier()

    def upper_forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_classifier(x)
        return x

    def lower_forward(self, x):
        x = self.feature_extractor(x)
        x = self.domain_classifier(x)
        return x


if __name__ == "__main__":
    domain_adversarial_network = DomainAdverseNetwork()
    x = torch.rand(4,1,32,32)
    print(domain_adversarial_network.upper_forward(x).shape)
    print(domain_adversarial_network.lower_forward(x).shape)
    for param in domain_adversarial_network.feature_extractor.parameters():
        print(param.shape)
