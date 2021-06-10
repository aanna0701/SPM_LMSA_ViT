#!/usr/bin/env python

import argparse
from time import time
from utils.mix import cutmix_data, mixup_data, mixup_criterion
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
from utils.losses import LabelSmoothingCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
from utils.cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar

best_acc1 = 0
input_size = 32


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path', default='/dataset', type=str, help='dataset path')
   
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100','IMNET'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=3e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='deit', choices=['deit', 'g-deit', 'vit', 'g-vit', 'pit', 'g-pit'])

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_true', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag')

    parser.add_argument('--seed', type=int, help='seed')
    
    # Augmentation parameters
    parser.add_argument('--aa', action='store_true', help='Auto augmentation used'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
  
    parser.add_argument('--cm',action='store_true' , help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu',action='store_true' , help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    # Autoaugmentation
    parser.add_argument('--rand_aug', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    
    parser.add_argument('--enable_rand_aug', action='store_true', help='Enabling randaugment')
    
    parser.add_argument('--enable_deit', action='store_true', help='Enabling randaugment')

    return parser


def main(args):
    global best_acc1    

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
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        
    elif args.dataset == 'IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('ImageNet')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 1000
        img_mean, img_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        img_size = 224
    
    '''
        Model 
    '''    

    if args.model == 'vit':
        model = m.make_ViT(args.depth, args.channel,GA=False, heads = args.heads, num_classes=n_classes)
        
    
    elif args.model == 'g-vit':
        model = m.make_ViT(args.depth, args.channel,GA=True, heads = args.heads, num_classes=n_classes)

    elif args.model == 'pit':
        model = m.P_ViT_conv(args.depth, num_classes=n_classes)
        
    
    elif args.model == 'g-pit':
        model = m.P_GiT_conv(args.depth, num_classes=n_classes)
        
    print(Fore.GREEN+'*'*80)
    logger.debug(f"Creating model: {model_name}")    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {n_parameters}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*'*80+Style.RESET_ALL)
    
    '''
        Criterion
    '''
    
    if args.ls:
        print(Fore.YELLOW + '*'*80)
        logger.debug('label smoothing used')
        print('*'*80+Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()
    
    else:
        criterion = nn.CrossEntropyLoss()
        
    '''
        GPU
    '''

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    
    '''
        Trainer
    '''

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 300, max_lr=args.lr, min_lr=5e-5, warmup_steps=args.warmup)
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]


    if args.cm:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Cutmix used')
        print('*'*80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Mixup used')
        print('*'*80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Autoaugmentation used')
        print('*'*80 + Style.RESET_ALL)
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            
            CIFAR10Policy()
        ]
        
    augmentations += [                
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        *normalize]
    
    
    augmentations = transforms.Compose(augmentations)

    '''
        Data Loader
    '''
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':

        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    '''
        Training
    '''
    
    summary(model, (3, 32, 32))
    
    print()
    print("Beginning training")
    print()
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_loader, model, criterion, args, get_lr(optimizer), epoch=epoch)
        logger_dict.print()
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            
        scheduler.step()        
        print(f'Best acc {best_acc1:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)

    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    mix = ''
    mix_paramter = 0
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                mix = 'cutmix'
                mix_paramter = args.beta        
                slicing_idx, y_a, y_b, lam = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]]
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)                
            else:
                output = model(images)
                loss = criterion(output, target)
        
        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                mix = 'mixup'
                mix_paramter = args.alpha
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            
            else:
                mix = 'none'
                mix_paramter = 0
                output = model(images)
                loss = criterion(output, target)
        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < 0.3:
                switching_prob = np.random.rand(1)
                
                # Cutmix
                if switching_prob < 0.5:
                    mix = 'cutmix'
                    mix_paramter = args.beta
                    slicing_idx, y_a, y_b, lam = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]]
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)         
                
                # Mixup
                else:
                    mix = 'mixup'
                    mix_paramter = args.alpha
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)                               
            
            else:
                mix = 'none'
                mix_paramter = 0
                output = model(images)
                loss = criterion(output, target)     
        
        # No Mix
        else:
            mix = 'none'
            mix_paramter = 0
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}][T][{i}] \t Loss: {avg_loss:.4e} \t Top-1: {avg_acc1:6.2f} \t\t LR: {get_lr(optimizer):.6f} \t\t Mix: {mix} ({mix_paramter})'+' '*10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)


def validate(val_loader, model, criterion, args, lr, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            
            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}] \t Loss: {avg_loss:.4e} \t Top-1: {avg_acc1:6.2f} \t\t LR: {lr:.6f}')
    print()        
    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    # total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(Fore.BLUE)
    print('*'*80)
    # logger.debug(f'[Epoch {epoch+1}] \t Top-1 {avg_acc1:6.2f} \t lr {lr:.6f} \t Time: {total_mins:.2f}')
    
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    return avg_acc1


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model + "-{}-{}-{}-{}-Seed{}".format(args.depth, args.heads, args.channel, args.tag, args.seed)
    save_path = os.path.join(os.getcwd(), 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # logger

    log_dir = os.path.join(save_path, 'logs.txt')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'w')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    
    main(args)
