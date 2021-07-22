#!/usr/bin/env python
from utils.autoaug import SVHNPolicy
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
import os
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from models.vit_pytorch.git import *
from utils.scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter

best_acc1 = 0
best_acc5 = 0
input_size = 32



def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'M-IMNET', 'SVHN', 'IMNET'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='deit', choices=['vit', 'g-vit', 'pit', 't2t-vit', 'cvt', 'res56', 'mobile2', 'resxt29', 'dense121', 'vgg16'])

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_true', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag')

    parser.add_argument('--seed', type=int, help='seed')

    parser.add_argument('--down_conv', action='store_true', help='down conv embedding')
    
    parser.add_argument('--sd', default=0, type=float, help='rate of stochastic depth')
    
    parser.add_argument('--ver', default=1, type=int, help='Version')
    
    parser.add_argument('--resume', default=False, help='Version')
    
    # Augmentation parameters
    parser.add_argument('--aa', action='store_true', help='Auto augmentation used'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # Mixup params
  
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
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation')
    
    # Random Erasing
    parser.add_argument('--re', default=0, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')

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
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        logger.debug('SVHN')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
        patch_size = 4
        in_channels = 3
        
    elif args.dataset == 'IMNET':
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        print(Fore.YELLOW+'*'*80)
        logger.debug('IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 1000
        img_mean, img_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        img_size = 224
        patch_size = 16
        in_channels = 3
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
        patch_size = 8
        in_channels = 3
        
    elif args.dataset == 'M-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('M-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 64
        img_mean, img_std = (0.4711, 0.4499, 0.4031), (0.2747, 0.2660, 0.2815)
        img_size = 84
        patch_size = 8
        in_channels = 3
    
    '''
        Model 
    '''    
    
    # ViTs
    
    dropout = False
    if args.dropout:
        dropout = args.dropout
    if args.model == 'vit':
        from models.vit_pytorch.vit import ViT        
        dim_head = args.channel // args.heads
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head, dropout=dropout, stochastic_depth=args.sd)
    #     model = m.make_ViT(args.depth, args.channel, down_conv=args.down_conv, dropout=dropout, GA=False, heads = args.heads, num_classes=n_classes, in_channels=in_channels, img_size=img_size)
        
    
    elif args.model == 'g-vit':
        from models.vit_pytorch.git import GiT        
        dim_head = args.channel // args.heads
        model = GiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head, dropout=dropout, stochastic_depth=args.sd)

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
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, mlp_dim=args.channel*2, depth=args.depth, heads=args.heads, dim_head=dim_head, dropout=dropout, stochastic_depth=args.sd)

    elif args.model =='t2t-vit':
        from models.vit_pytorch.t2t import T2TViT
        model = T2TViT(image_size=img_size, num_classes=n_classes, depth=args.depth)
        

    elif args.model =='cvt':
        from models.vit_pytorch.cvt import CvT
        model = CvT(num_classes=n_classes)
        
    # Convnets

    elif args.model == 'vgg16':
        from models.conv_cifar_pytoch.vgg import VGG
        model = VGG('VGG16')

    elif args.model == 'res56':
        from models.conv_cifar_pytoch.resnet import resnet56
        model = resnet56()

    elif args.model == 'resxt29':
        from models.conv_cifar_pytoch.resnext import ResNeXt29_32x4d
        model = ResNeXt29_32x4d()

    elif args.model == 'mobile2':            
        from models.conv_cifar_pytoch.mobilenetv2 import MobileNetV2
        model = MobileNetV2()

    elif args.model == 'dense121':
        from models.conv_cifar_pytoch.densenet import DenseNet121
        model = DenseNet121()
    
    # elif args.model == 'g-pit':
    #     model = m.P_GiT_conv(args.channel, num_classes=n_classes, dropout=dropout, in_channels=in_channels, img_size=img_size, down_conv=args.down_conv)
        
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
        
    if args.sd > 0.:
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*'*80+Style.RESET_ALL)         
        
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

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]


    if args.cm:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Cutmix used')
        print('*'*80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Mixup used')
        print('*'*80 + Style.RESET_ALL)
    if args.ra > 1:        
        from utils.sampler import RASampler
        print(Fore.YELLOW+'*'*80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*'*80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Autoaugmentation used')      
        
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [
                
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy()
            ]
            
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")    
            from utils.autoaug import SVHNPolicy
            augmentations += [
                
              transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                SVHNPolicy()
            ]
        
            
        elif args.dataset == 'IMNET':
            print("ImageNet Policy")    
            from utils.autoaug import ImageNetPolicy
            augmentations += [
                transforms.RandomResizedCrop(224),
                ImageNetPolicy()
            ]
            
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [                
              transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy()
            ]
            
        print('*'*80 + Style.RESET_ALL)
        

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*'*80+Style.RESET_ALL)    
        
        
        augmentations += [                
            transforms.ToTensor(),
            *normalize,
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=img_mean)]
    
    else:
        augmentations += [                
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
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
        
        
    elif args.dataset == 'SVHN':

        train_dataset = datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'imnet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'imnet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size), transforms.ToTensor(), *normalize]))
        
    elif args.dataset == 'T-IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))
        
    elif args.dataset == 'M-IMNET':
    
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'mini_imagenet_84', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'mini_imagenet_84', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, args.ra, 3, shuffle=True, drop_last=False))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, 300, max_lr=args.lr, min_lr=min_lr, warmup_steps=args.warmup)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    
    summary(model, (3, img_size, img_size))
    # print(model)
    
    print()
    print("Beginning training")
    print()
    
    lr = optimizer.param_groups[0]["lr"]
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = checkpoint['loss']
        scheduler = checkpoint['scheduler']
        args.epochs = checkpoint['epoch'] + 1
    
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1, acc5 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lr,
            'scheduler': scheduler.state_dict(), 
            }, 
            os.path.join(save_path, 'checkpoint.pth'))
        logger_dict.print()
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
        
        if acc5 > best_acc5:
            best_acc5 = acc5
            
        
        print(f'Best acc1 {best_acc1:.2f}, Best acc5 {best_acc5:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)        
        
        writer.add_scalar("Learning Rate", lr, epoch)
        
        # for i in range(len(model.transformer.scale)):
        #     for idx, scale in enumerate(model.transformer.scale[str(i)]):
                
        #         writer.add_scalar(f"Scale/depth{i}_head{idx}", nn.functional.sigmoid(scale), epoch)
        
    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, best top-5: {best_acc5:.2f}, final top-1: {acc1:.2f}, final top-5: {acc5:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler,  args):
    model.train()
    loss_val, acc1_val, acc5_val = 0, 0, 0
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
                mix = 'none'
                mix_paramter = 0
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
            if r < args.mix_prob:
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

        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1, avg_acc5 = (loss_val / n), (acc1_val / n), (acc5_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}   Mix: {mix} ({mix_paramter})'+' '*10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)

    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val, acc5_val = 0, 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            
            output = model(images)
            loss = criterion(output, target)
            
            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            acc5 = acc[1]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))
            acc5_val += float(acc5[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1, avg_acc5 = (loss_val / n), (acc1_val / n), (acc5_val / n)
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   Top-5: {avg_acc5:6.2f}   LR: {lr:.6f}')
    print()        

    # total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(Fore.BLUE)
    print('*'*80)
    # logger.debug(f'[Epoch {epoch+1}] \t Top-1 {avg_acc1:6.2f} \t lr {lr:.6f} \t Time: {total_mins:.2f}')
    
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    logger_dict.update(keys[4], avg_acc5)
    
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)
    
    return avg_acc1, avg_acc5


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model + f"-{args.depth}-{args.heads}-{args.channel}-{args.dataset}-{args.tag}-Seed{args.seed}"
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
