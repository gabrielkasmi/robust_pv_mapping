# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
sys.path.append('../')

from torchvision.models import resnet50
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from data_loader import get_loader
from utils import accuracy, Tracker
from coral import coral

import json
import torchvision
from torch.utils.data import DataLoader
import os
from src import bdappv


# adapted from https://github.com/DenisDsh/PyTorch-Deep-CORAL/tree/master

# hard coded parameters
DATASET_DIR="../../../data/bdappv"
IMAGES_LIST_DIR="../data"


def setup_datasets(images_list, dataset_dir, batch_size):
    """
    sets up the training and testing datasets for the use
    case on BDAPPV
    """

    # Data preparation: train and test instances
    # focus on the sensitivty to acquisition conditoins

    # baseline transforms: no corruptions
    baseline = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
    ])

    datasets = {}

    for case in ['google', 'ign']:

        # test instances: test source and test target

        path = os.path.join(dataset_dir, case)

        if case == "google":

            dataset = bdappv.BDAPPVClassification(path, size = 200, transform=baseline, images_list=images_list["test"], random = False, downsample=200)
        else:
            dataset = bdappv.BDAPPVClassification(path, size = 200, images_list=images_list["test"], random = False, transform = baseline)

        database = DataLoader(dataset, batch_size=batch_size)
        datasets[case] = database

        # train instances: Google train/IGN train
        case_name='train_{}'.format(case)
        if case=="google":
            dataset = bdappv.BDAPPVClassification(path, size = 200, transform=baseline, images_list=images_list["train"], random = False, downsample=200)
        else:
            dataset = bdappv.BDAPPVClassification(path, size = 200, images_list=images_list["train"], random = False, transform = baseline)

        database=DataLoader(dataset,batch_size=batch_size)
        datasets[case_name]=database

    return datasets 


def train(model, optimizer, source_loader, target_loader, tracker, args, epoch=0):

    model.train()
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

    # Trackers to monitor classification and CORAL loss
    classification_loss_tracker = tracker.track('classification_loss', tracker_class(**tracker_params))
    coral_loss_tracker = tracker.track('CORAL_loss', tracker_class(**tracker_params))

    min_n_batches = min(len(source_loader), len(target_loader))

    tq = tqdm(range(min_n_batches), desc='{} E{:03d}'.format('Training + Adaptation', epoch), ncols=0)

    for _ in tq:

        source_data, source_label, _ = next(iter(source_loader))
        target_data, _, _ = next(iter(target_loader))  # Unsupervised Domain Adaptation

        source_data, source_label = Variable(source_data.to(device=args.device)), Variable(source_label.to(device=args.device))
        target_data = Variable(target_data.to(device=args.device))

        optimizer.zero_grad()

        out_source = model(source_data)
        out_target = model(target_data)

        classification_loss = F.cross_entropy(out_source, source_label)

        # This is where the magic happens
        coral_loss = coral(out_source, out_target)
        composite_loss = classification_loss + float(args.lambda_coral) * coral_loss

        composite_loss.backward()
        optimizer.step()

        classification_loss_tracker.append(classification_loss.item())
        coral_loss_tracker.append(coral_loss.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(classification_loss=fmt(classification_loss_tracker.mean.value),
                       coral_loss=fmt(coral_loss_tracker.mean.value))


def evaluate(model, data_loader, dataset_name, tracker, args, epoch=0):
    model.eval()

    tracker_class, tracker_params = tracker.MeanMonitor, {}
    acc_tracker = tracker.track('{}_accuracy'.format(dataset_name), tracker_class(**tracker_params))

    loader = tqdm(data_loader, desc='{} E{:03d}'.format('Evaluating on %s' % dataset_name, epoch), ncols=0)

    accuracies = []
    with torch.no_grad():
        for target_data, target_label, _ in loader:
            target_data = Variable(target_data.to(device=args.device))
            target_label = Variable(target_label.to(device=args.device))

            output = model(target_data)

            accuracies.append(accuracy(output, target_label))

            acc_tracker.append(sum(accuracies)/len(accuracies))
            fmt = '{:.4f}'.format
            loader.set_postfix(accuracy=fmt(acc_tracker.mean.value))


def main():

    # Paper: In the training phase, we set the batch size to 128,
    # base learning rate to 10−3, weight decay to 5×10−4, and momentum to 0.9

    parser = argparse.ArgumentParser(description='Train - Evaluate DeepCORAL model')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--decay', default=5e-4,
                        help='Decay of the learning rate')
    parser.add_argument('--momentum', default=0.95,
                        help="Optimizer's momentum")
    parser.add_argument('--lambda_coral', type=float, default=0.5,
                        help="Weight that trades off the adaptation with "
                             "classification accuracy on the source domain")
    parser.add_argument('--source', default='google',
                        help="Source Domain (dataset)")
    parser.add_argument('--target', default='ign',
                        help="Target Domain (dataset)")

    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    
    # replace here with IGN and Google

    images_list=json.load(open(os.path.join(IMAGES_LIST_DIR, 'images_lists.json')))
    datasets=setup_datasets(images_list, DATASET_DIR, int(args.batch_size))

    source_train_loader=datasets['train_google']
    target_train_loader=datasets['train_ign']

    source_evaluate_loader=datasets['google']
    target_evaluate_loader=datasets['ign']

    print('Dataset setup complete')

    n_classes = len(source_train_loader.dataset)

    # ~ Paper : "We initialized the other layers with the parameters pre-trained on ImageNet"
    # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    # change for a resnet
    model = resnet50(pretrained=True)
    # ~ Paper : The dimension of last fully connected layer (fc8) was set to the number of categories (31)

    # change the FC Layer to match the requirements of a ResNet
    model.fc = nn.Linear(2048, n_classes)

    # ~ Paper : and initialized with N(0, 0.005)
    torch.nn.init.normal_(model.fc.weight, mean=0, std=5e-3)

    # Initialize bias to small constant number (http://cs231n.github.io/neural-networks-2/#init)
    model.fc.bias.data.fill_(0.01)

    model = model.to(device=args.device)

#     # ~ Paper : "The learning rate of fc8 is set to 10 times the other layers as it was training from scratch."
#     optimizer = torch.optim.SGD([
#         {'params':  model.features.parameters()},
#         {'params': model.fc.parameters()},
#         # fc8 -> 7th element (index 6) in the Sequential block
#         {'params': model.fc.parameters(), 'lr': 10 * args.lr}
#     ], lr=args.lr, momentum=args.momentum)  # if not specified, the default lr is used

    # Setup the optimizer
    optimizer = torch.optim.SGD([
        {'params': model.conv1.parameters()},  # First convolutional layer
        {'params': model.layer1.parameters()},  # First residual block
        {'params': model.layer2.parameters()},  # Second residual block
        {'params': model.layer3.parameters()},  # Third residual block
        {'params': model.layer4.parameters()},  # Fourth residual block
        {'params': model.avgpool.parameters()},  # Average pooling layer
        {'params': model.fc.parameters(), 'lr': 10 * float(args.lr)}  # FC layer (last layer)
    ], lr=float(args.lr), momentum=float(args.momentum))

    tracker = Tracker()

    for i in range(int(args.epochs)):
        train(model, optimizer, source_train_loader, target_train_loader, tracker, args, i)
        evaluate(model, source_evaluate_loader, 'source', tracker, args, i)
        evaluate(model, target_evaluate_loader, 'target', tracker, args, i)

    # Save logged classification loss, coral loss, source accuracy, target accuracy
    torch.save(tracker.to_dict(), "log.pth")

    # save the model
    torch.save(model.state_dict(), "model_weights.pth")
    print('Stats and model saved')


if __name__ == '__main__':
    main()