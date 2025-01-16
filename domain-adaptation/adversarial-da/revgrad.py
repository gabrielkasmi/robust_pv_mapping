# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse


import sys
sys.path.append('../../')

# from torchvision.models import resnet50 - not used for now
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse


import json
import torchvision
from torch.utils.data import DataLoader
import os
from src import bdappv


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from data import MNISTM
from models import Net
from utils import GrayscaleToRgb, GradientReversal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hard coded parameters
DATASET_DIR="../../../../data/bdappv"
IMAGES_LIST_DIR="../../data"

images_list=json.load(open(os.path.join(IMAGES_LIST_DIR, 'images_lists.json')))

# BDAPPV dataset loader
def setup_datasets(images_list, dataset_dir):
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

        #database = DataLoader(dataset, batch_size=batch_size)
        datasets[case] = dataset

        # train instances: Google train/IGN train
        case_name='train_{}'.format(case)
        if case=="google":
            dataset = bdappv.BDAPPVClassification(path, size = 200, transform=baseline, images_list=images_list["train"], random = False, downsample=200)
        else:
            dataset = bdappv.BDAPPVClassification(path, size = 200, images_list=images_list["train"], random = False, transform = baseline)

        #database=DataLoader(dataset,batch_size=batch_size)
        datasets[case_name]=dataset

    return datasets 


def main(args):
    model = Net().to(device)
    model.load_state_dict(torch.load(args.MODEL_FILE))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    discriminator = nn.Sequential( # Same structure for the discriminator
        nn.Linear(2048, args.batch_size),
        nn.ReLU(),
        nn.Linear(args.batch_size, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    datasets=setup_datasets(images_list, DATASET_DIR)

    source_dataset=datasets['train_google']
    target_dataset=datasets['ign']

    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    
    # target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels, _), (target_x, _, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                features = feature_extractor(x).view(x.shape[0], -1)
                domain_preds = discriminator(features).squeeze()
                label_preds = clf(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        torch.save(model.state_dict(), 'trained_models/revgrad.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=15)
    args = arg_parser.parse_args()
    main(args)
