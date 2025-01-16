# -*- coding: utf-8 -*-

# trains a model using the scattering transform
# as a feature extractor.

# Libraries

import os
import json
import torch, torchvision
from kymatio.torch import Scattering2D
import torch.nn as nn
from src.bdappv import BDAPPVClassification
import argparse
from src.utils import confusion
from PIL import ImageFile

parser = argparse.ArgumentParser(description = 'Training the scattering transform')

parser.add_argument('--J', default = 3, help = "Number of levels",type=int)
parser.add_argument('--batch_size', default = 256, help = "Batch size")
parser.add_argument('--device', default = 'cuda', help = "Device for training")
parser.add_argument('--dataset_dir', default = "../../data/bdappv", help = "location of the training images")
parser.add_argument('--target_dir', default = "scattering", help = "location of the outputs")
parser.add_argument('--input_shape', default = 200, help = "size of the input images")
parser.add_argument('--model_name',default="model_ign_", help = "name of the model (or cases)")
parser.add_argument('--images_list',default=True, help = "fix the training and testing datasets")
parser.add_argument('--n_epochs',default=5, help = "Number of epochs")


args = parser.parse_args()

dataset_dir = args.dataset_dir
target_dir=args.target_dir

batch_size = args.batch_size
device=args.device
input_shape=args.input_shape
J=args.J
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_K(L,J,num_channels=3):
    """
    L:number of angles of the scattering transform
    J: number of scales
    num_channels: number of input channels
    """
    return int(1 + L*J + (L**2)*J*((J-1)/2))*num_channels    

class Scattering2dCNN(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG
    '''
    def __init__(self, J, \
                 input_shape,\
                 classifier_type='linear',\
                 L=8, \
                 num_classes=2):
        
        super(Scattering2dCNN, self).__init__()
        self.in_channels = get_K(L,J)
        self.J=J
        self.input_shape=input_shape
        self.classifier_type = classifier_type
        self.num_classes=num_classes
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.in_channels
        self.out_shape=self.input_shape // 2**self.J

        self.bn = nn.BatchNorm2d(self.K)
        if self.classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    self.in_channels = v

            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier =  nn.Linear(1024*4, self.num_classes)

        elif self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                        nn.Linear(self.K*self.out_shape*self.out_shape, 1024), nn.ReLU(),
                        nn.Linear(1024, 1024), nn.ReLU(),
                        nn.Linear(1024, 10))
            self.features = None

        elif self.classifier_type == 'linear':
            self.classifier = nn.Linear(self.K*self.out_shape*self.out_shape,self.num_classes)
            self.features = None


    def forward(self, x):
        x = self.bn(x.view(-1, self.K, self.out_shape, self.out_shape))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch, scattering, criterion):

    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        pred=nn.functional.softmax(output, dim=1)
        pred=pred[:,1] # take the value predicted for the 2nd column
        loss=criterion(pred, target.float())
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, scattering, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    tp, tn, fp, fn = 0, 0, 0, 0

    i=0

    with torch.no_grad():
        for data, target, _ in test_loader:
            i+=1
            #if i == 4:
            #    break
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            preds=nn.functional.softmax(output, dim=1)
            preds=preds[:,1] # take the value predicted for the 2nd column
            test_loss += criterion(preds, target.float()).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            true_positives, false_positives, \
                true_negatives, false_negatives, _ = confusion(pred, target)
            
            tp += true_positives
            tn += true_negatives
            fp += false_positives
            fn += false_negatives

    test_loss /= len(test_loader.dataset)

    precision= tp/(tp+fp)
    recall=tp/(tp+fn)

    f1_score=2*precision*recall / (precision+recall)

    print('\nTest set: Average loss: {:.4f}, F1: {:0.2f}\n'.format(
        test_loss, f1_score))
    
    return test_loss

def save_model(model, target_dir, name):
    """
    saves the model if it achieves the lowest loss on the test set
    """
    model_name = 'model_{}.pth'.format(name)
    torch.save(model, os.path.join(target_dir, model_name))

if args.images_list:
    images_list = json.load(open("data/images_lists.json"))

scattering=Scattering2D(J=J, shape=(input_shape,input_shape))
scattering = scattering.to(device)
model=Scattering2dCNN(J,input_shape).to(device)

# baseline transforms: no corruptions
BASELINE = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor()#,
    #torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])

datasets = {
    'ign_train': BDAPPVClassification(os.path.join(dataset_dir, "ign"), size = input_shape, \
                                                    transform=BASELINE, images_list=images_list["train"], \
                                                        random = False),
    'ign_test'    : BDAPPVClassification(os.path.join(dataset_dir, "ign"), size = input_shape, \
                                                    transform=BASELINE, images_list=images_list["test"], \
                                                        random = False),
} 


pin_memory=True #True if we use cuda.
num_workers=4

criterion = nn.BCELoss()

# Optimizer
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


destination_name=args.model_name + str(J)
destination_dir=os.path.join(target_dir,destination_name)

if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

def main():

    print('Initialization')

    train_loader = torch.utils.data.DataLoader(datasets['ign_train'], batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(datasets['ign_test'], batch_size=batch_size)

    print('Setting up the loss')

    loss=test(model, device, test_loader, scattering, criterion)
    tmp_loss=loss    
    save_model(model,destination_dir,0) # save tghe initial model (with random weights)
    
    for epoch in range(0, args.n_epochs):

        train(model, device, train_loader, optimizer, epoch+1, scattering, criterion)

        loss=test(model, device, test_loader, scattering, criterion)

        if loss<tmp_loss:
            save_model(model,destination_dir,epoch+1)
            tmp_loss=loss


if __name__ == '__main__':
    main()