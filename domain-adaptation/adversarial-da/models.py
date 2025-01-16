# from torch import nn
# 
# 
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 10, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(10, 20, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(),
#         )
#         
#         self.classifier = nn.Sequential(
#             nn.Linear(320, 50),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(50, 10),
#         )
# 
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits

import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet-50 as the feature extractor
        resnet = models.resnet50(pretrained=True)
        
        # Remove the fully connected layer of ResNet-50
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Output: [batch_size, 2048, 1, 1]
        
        # Define the classifier for two output classes
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # Map ResNet-50's output to a smaller dimension
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Two output classes
        )

    def forward(self, x):
        # Extract features using ResNet-50
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)  # Flatten the output: [batch_size, 2048]
        
        # Pass the features through the classifier
        logits = self.classifier(features)
        return logits