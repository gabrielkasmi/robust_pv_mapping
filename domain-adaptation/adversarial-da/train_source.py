import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from models import Net
from utils import GrayscaleToRgb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# own
import sys
sys.path.append('../../')
import json
import torchvision
from torch.utils.data import DataLoader
import os
from src import bdappv

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

def create_dataloaders(datasets,batch_size):

    # instandiate the datasets

    train_loader=DataLoader(datasets['train_google'],batch_size=batch_size)
    val_loader=DataLoader(datasets['google'], batch_size=batch_size)

    return train_loader, val_loader

# def create_dataloaders(batch_size):
#     dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
#                     transform=Compose([GrayscaleToRgb(), ToTensor()]))
#     shuffled_indices = np.random.permutation(len(dataset))
#     train_idx = shuffled_indices[:int(0.8*len(dataset))]
#     val_idx = shuffled_indices[int(0.8*len(dataset)):]
# 
#     train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
#                               sampler=SubsetRandomSampler(train_idx),
#                               num_workers=1, pin_memory=True)
#     val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
#                             sampler=SubsetRandomSampler(val_idx),
#                             num_workers=1, pin_memory=True)
#     return train_loader, val_loader
# 

def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true, _ in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):

    datasets=setup_datasets(images_list, DATASET_DIR)

    train_loader, val_loader = create_dataloaders(datasets,args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on Google BDAPPV')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=10)
    args = arg_parser.parse_args()
    main(args)
