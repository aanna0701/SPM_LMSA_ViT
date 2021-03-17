import os
from glob import glob
import sys
from colorama import init, Fore, Style
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from training_functions import EarlyStopping
import logging as log
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

import models

########### constants

init(autoreset=True)
use_cuda = True
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
now = datetime.now().strftime('%Y-%m-%d-%H_%M')

############ varaiables

FINETUNING = False
log_interval = 100
batch_size = 128
test_batch_size = 128
epochs = 200
ealry_stopping_patience = 500

############ args

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--dataset_dir', help='path of input images', default='/media/CVIP/Hyundai2020/dataset/training/0809')
parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
parser.add_argument('--model', help='model', required=True)
parser.add_argument('--multi_gpus', help='multi gpus', action='store_true')
parser.add_argument('--seed', help='seed', type=int, required=True)

args = parser.parse_args()

assert args.model in ['resnet56', 'nlb_1', 'nlb_2', 'nlb_3', 'nlb_4', 'nlb_5', 'nlb_6', 'nlb_9', 'gasa'], 'Unexpected model!'

if __name__ == "__main__":

    ############ random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ############ save path

    save_path = os.path.join(os.getcwd(), "save")
    save_path = os.path.join(save_path, now + '_' + args.model + "_seed" + str(args.seed))   
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    ############ logger

    log_dir = os.path.join(save_path, 'log.txt')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('[%(asctime)s]\n%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir)
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    ############ training objects

    writer = SummaryWriter()
    early_stopping = EarlyStopping(patience=ealry_stopping_patience, verbose=1, mode='max')

    ############ data loaders

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, 4),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                         ])) ,
                         batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                         ])),
                         batch_size=batch_size, shuffle=True)


    # logger.debug(Fore.RED + Style.BRIGHT + 
    #              '\n# Number of training data : {}\n# Number of validation data : {}\n'.format(len(train_loader)*batch_size, len(test_loader)*test_batch_size)
    #              + Style.RESET_ALL)

    ############ model load

    if args.model == 'resnet56':
        model = models.resnet56()
            
    elif args.model == 'nlb_1':
        model = models.resnet56_nlb_1()
    
    elif args.model == 'nlb_2':
        model = models.resnet56_nlb_2()
    
    elif args.model == 'nlb_3':
        model = models.resnet56_nlb_3()

    elif args.model == 'nlb_4':
        model = models.resnet56_nlb_4()

    elif args.model == 'nlb_5':
        model = models.resnet56_nlb_5()
    
    elif args.model == 'nlb_6':
        model = models.resnet56_nlb_6()
    
    elif args.model == 'nlb_9':
        model = models.resnet56_nlb_9()
        
    logger.debug(Fore.MAGENTA + Style.BRIGHT + '\n# Model: {}\
        \n# Initial Learning Rate: {}\
            \n# Seed: {}'.format(args.model, args.lr, args.seed) + Style.RESET_ALL)  
        
    if args.multi_gpus: # Using multi-gpu
        model = nn.DataParallel(model)
        print(Fore.RED + Style.BRIGHT + '\n# Multi Gpus Used!!' + Style.RESET_ALL)  
    
    model.cuda()
    summary(model, (3, 32, 32))

    ############ trainers

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001 )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 175], gamma=0.1)

    ############ training loop

    for epoch in tqdm(range(1, epochs+1)):
        # Train Mode
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()   # backpropagation 계산 전 opimizer 초기화
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()     # backpropagation 수행
            optimizer.step()    # weight update

            if batch_idx % log_interval == 0:
                logger.debug('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Test Mode
        model.eval()    # batch norm이나 droput 등을 train mode로 변환
        test_loss = 0
        correct = 0
        with torch.no_grad():   # autograd engine, backpropagation이나 gradien 계산등을 꺼서 memory usage를 줄이고 속도 향상
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()     # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)       # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuray = 100. * correct / len(test_loader.dataset)

        logger.debug(Fore.BLUE + Style.BRIGHT + '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                test_accuray))

        # early stopping
        early_stop = early_stopping.validate(test_accuray)

        if not early_stop:
            if early_stopping.best_value <= test_accuray:
                logger.debug(Fore.GREEN + Style.BRIGHT + 'best model updates!!!!!\n')
                # model.state_dict(): 딕셔너리 형태로 모델의 Layer와 Weight가 저장되어있음.
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path+'/best.pt')

        else:
            logger.debug(Fore.RED + Style.BRIGHT + 'best acc: {}'.format(early_stop))
            # model.state_dict(): 딕셔너리 형태로 모델의 Layer와 Weight가 저장되어있음.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path+'/final.pt')
            sys.exit()

        # Learning Rate Schedule
        scheduler.step()

        writer.add_scalar('Loss/train/', loss, epoch)    
        writer.add_scalar('Loss/test/', test_loss, epoch)
        writer.add_scalar('Accuracy/test/', test_accuray, epoch)
        writer.add_scalar('Learning Rate/', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Initial Learning Rate/', args.lr, epoch)

