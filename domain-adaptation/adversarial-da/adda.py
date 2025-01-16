# main code for implementing ADDA for implementiing the DA methods of pytorch-DA

# -*- coding: utf-8 -*-
#!/usr/bin/env python

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


# library imports from ADDA script 
# from torchvision.datasets import MNIST
from tqdm import tqdm, trange

# from data import MNISTM
from models import Net
from utils import loop_iterable, set_requires_grad # , GrayscaleToRgb


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
    source_model = Net().to(device)
    source_model.load_state_dict(torch.load(args.MODEL_FILE))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    clf = source_model
    source_model = source_model.feature_extractor

    target_model = Net().to(device) # Same structure, small convnet for 
    target_model.load_state_dict(torch.load(args.MODEL_FILE))
    target_model = target_model.feature_extractor

    discriminator = nn.Sequential( # Same structure for the discriminator
        nn.Linear(2048, args.batch_size),
        nn.ReLU(),
        nn.Linear(args.batch_size, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2

    # modify here the datasets
    #source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
    #                      transform=Compose([GrayscaleToRgb(), ToTensor()]))
    
    datasets=setup_datasets(images_list, DATASET_DIR)

    source_dataset=datasets['train_google']
    target_dataset=datasets['ign']

    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    
    # target_dataset = MNISTM(train=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters())
    target_optim = torch.optim.Adam(target_model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)

            for z in range(args.k_clf):
                _, (target_x, _, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()

                loss = criterion(preds, discriminator_y)
                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations*args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        clf.feature_extractor = target_model
        torch.save(clf.state_dict(), 'trained_models/adda.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--iterations', type=int, default=100)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=10)
    args = arg_parser.parse_args()
    main(args)
