#!/usr/bin/env python
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import Fore, Style
from torchsummary import summary
import os
from utils.relative_norm_residuals import compute_rank
from utils.logger_dict import Logger_dict
import argparse
from torch.utils.tensorboard import SummaryWriter

best_acc1 = 0
best_acc5 = 0
input_size = 32



def init_parser():
    parser = argparse.ArgumentParser(description='Rank collapse')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'M-IMNET', 'SVHN'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    parser.add_argument('-b', '--batch-size', default=1000, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')

    parser.add_argument('--model', type=str, default='deit', choices=['vit', 'g-vit', 'pit', 't2t-vit', 'cvt', 'res56', 'mobile2', 'resxt29', 'dense121', 'vgg16'])

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag')

    parser.add_argument('--down_conv', action='store_true', help='down conv embedding')

    parser.add_argument('--weights', type=str, required=True)
    
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    

    return parser


def main(args):
    global best_acc1    
    global best_acc5    

    '''
        Dataset
    '''
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
        in_channels = 3
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        in_channels = 3
        
    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        logger.debug('SVHN')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
        in_channels = 3
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
        in_channels = 3
        
    elif args.dataset == 'M-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('M-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 64
        img_mean, img_std = (0.4711, 0.4499, 0.4031), (0.2747, 0.2660, 0.2815)
        img_size = 84
        in_channels = 3
    
    '''
        Model 
    '''    
    
    # ViTs
    
    if args.model == 'vit':
        from models.vit_pytorch.vit import ViT        
        dim_head = args.channel // args.heads
        model = ViT(img_size=img_size, patch_size = 4, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head)
    #     model = m.make_ViT(args.depth, args.channel, down_conv=args.down_conv, GA=False, heads = args.heads, num_classes=n_classes, in_channels=in_channels, img_size=img_size)
        
    
    elif args.model == 'g-vit':
        from models.vit_pytorch.git import GiT        
        dim_head = args.channel // args.heads
        model = GiT(img_size=img_size, patch_size = 4, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head)

    elif args.model == 'pit':
        from models.vit_pytorch.pit import PiT
        if img_size == 32:
            patch_size = 4
        elif img_size > 32:
            patch_size = 8
        dim_head = args.channel // args.heads
        if args.channel == 144:
            args.channel = 64
        else:
            args.channel = 96
        args.heads = 2
        args.depth = (2, 6, 4)
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head)

    elif args.model =='t2t-vit':
        from models.vit_pytorch.t2t import T2TViT
        model = T2TViT(image_size=img_size, num_classes=n_classes, depth=args.depth)
        

    elif args.model =='cvt':
        from models.vit_pytorch.cvt import CvT
        model = CvT(num_classes=n_classes)
        
    '''
        GPU
    '''

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        
    
    '''
        Trainer
    '''

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    '''
        Data Loader
    '''
    if args.dataset == 'CIFAR10':
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
        
    elif args.dataset == 'SVHN':
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'T-IMNET':
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))
        
    elif args.dataset == 'M-IMNET':
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'mini_imagenet_84', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    
    summary(model, (3, img_size, img_size))
    
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), args.weights, 'best.pth')))
    
    
    rank(val_loader, model, args)
    
    
def rank(val_loader, model, args):
    
    value = {}
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i < 1:
                if (not args.no_cuda) and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)

            
                _ = model(images)
                
                hidden_states = model.transformer.hidden_states
                if i == 0:
                    for key in hidden_states:
                        # value[key] = compute_rank(hidden_states[key]) 
                        value[key] = [hidden_states[key]]
                
                else:
                    for key in hidden_states:
                        # value[key] += compute_rank(hidden_states[key])
                        value[key].append(hidden_states[key])
        
        
        for key in value:
            value[key] = torch.cat(value[key], dim=0)
            value[key] = compute_rank(value[key])
        
        print('done')
        print(f'{value}')
        
       


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    
    model_name = args.model + f"-{args.depth}-{args.heads}-{args.channel}-{args.dataset}-{args.tag}"
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1', 'V Top-5']
    
    main(args)
