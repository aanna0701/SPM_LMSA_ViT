#!/usr/bin/env python

import argparse
from time import time
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import init, Fore, Back, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
import models.create_model as m

best_acc1 = 0

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

    parser.add_argument('--model', type=str, default='deit', choices=['deit', 'g-deit', 'vit', 'g-vit'])

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')

    parser.add_argument('--label_smoothing', action='store_true', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag')

    parser.add_argument('--seed', type=int, help='seed')
    
    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--enable_mix', action='store_true', help='Enabling mixup')

    # Autoaugmentation
    parser.add_argument('--rand_aug', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    
    parser.add_argument('--enable_rand_aug', action='store_true', help='Enabling randaugment')

    return parser


def main(args):
    global best_acc1

    '''
        Dataset
    '''
    if args.dataset == 'CIFAR10':
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
        
    elif args.dataset == 'CIFAR100':
        n_classes = 100
        img_mean, img_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        
    elif args.dataset == 'IMNET':
        n_classes = 1000
        img_mean, img_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        img_size = 224
    
    '''
        Model 
    '''    
    if args.model == 'deit':
        model = m.make_ViT(args.depth, args.channel, heads = args.heads, num_classes=n_classes)

    elif args.model == 'g-deit':
        model = m.make_ViT(args.depth, args.channel,GA=True, heads = args.heads, num_classes=n_classes)

    elif args.model == 'vit':
        model = m.make_ViT(args.depth, args.channel,GA=False, heads = args.heads, num_classes=n_classes)
        args.disable_aug = True
    
    elif args.model == 'g-vit':
        model = m.make_ViT(args.depth, args.channel,GA=True, heads = args.heads, num_classes=n_classes)
        args.disable_aug = True
        
    print(Fore.GREEN+'*'*80)
    logger.debug(f"  Creating model: {model_name}  ")    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'  Number of params: {n_parameters}  ')
    logger.debug(f'  Initial learning rate: {args.lr:.6f}  ')
    logger.debug(f"  Start training for {args.epochs} epochs  ")
    print('*'*80+Style.RESET_ALL)
    
    '''
        Criterion
    '''
    
    if args.label_smoothing:
        print(Fore.YELLOW + '*'*80)
        print('label smoothing used')
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
        
    summary(model, (3, 32, 32))
    
    '''
        Trainer
    '''

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 300, max_lr=args.lr, min_lr=5e-5, warmup_steps=args.warmup)
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    '''
        Data Augmentation
    '''
    augmentations = []
    if args.enable_aug:
        print(Fore.YELLOW+'*'*80)
        print('Autoaugmentation used')
        print('*'*80 + Style.RESET_ALL)
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]
    elif args.enable_rand_aug:
        print(Fore.YELLOW+'*'*80)
        print('Randaugmentation used')
        print('*'*80 + Style.RESET_ALL)
        augmentations += [
            rand_augment_transform(config_str=args.rand_aug, hparams={'img_mean': img_mean})]
    augmentations += [                
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        *normalize,
    ]
    augmentations = transforms.Compose(augmentations)
    '''
        Mixup
    '''
    mixup_fn = None
    if args.enable_mix:
        print(Fore.YELLOW+'*'*80)
        print('Mixup used')
        print('*'*80 + Style.RESET_ALL)
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.1, num_classes=args.n_classes)
    '''
        Data Loader
    '''
    train_dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=augmentations)
    val_dataset = datasets.CIFAR10(
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
    print()
    print("Beginning training")
    print()
    time_begin = time()
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        cls_train(train_loader, model, criterion, optimizer, epoch, args, mixup_fn)
        acc1 = cls_validate(val_loader, model, criterion, args, get_lr(optimizer), epoch=epoch, time_begin=time_begin)
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            logger.debug('Best model update')
            
        scheduler.step()
        logger.debug(f'Best acc {best_acc1:.2f}')
        print('*'*80+Style.RESET_ALL)

    total_mins = (time() - time_begin) / 60
    print(Fore.RED+'*'*80)
    logger.debug(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'final top-1: {acc1:.2f}')
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


def cls_train(train_loader, model, criterion, optimizer, epoch, args, mixup_fn=None):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        if mixup_fn:
            images, target = mixup_fn(images, target)
        
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
            logger.debug(f'[Epoch {epoch+1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f} \t LR {get_lr(optimizer):.6f}')


def cls_validate(val_loader, model, criterion, args, lr, epoch=None, time_begin=None):
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
                logger.debug(f'[Epoch {epoch+1}][Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(Fore.BLUE+'*'*80)
    logger.debug(f'[Epoch {epoch+1}] \t Top-1 {avg_acc1:6.2f} \t lr {lr:.6f} \t Time: {total_mins:.2f}')
    

    return avg_acc1


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
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
    
    main(args)
