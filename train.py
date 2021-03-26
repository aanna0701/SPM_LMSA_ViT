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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

########### constants

init(autoreset=True)
use_cuda = True
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
now = datetime.now().strftime('%Y-%m-%d-%H_%M')

############ args

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--dataset_dir', help='path of input images', default='/media/CVIP/Hyundai2020/dataset/training/0809')
parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
parser.add_argument('--model', help='model', required=True)
parser.add_argument('--multi_gpus', help='multi gpus', action='store_true')
parser.add_argument('--seed', help='seed', type=int, required=True)
parser.add_argument('--n_blocks', help='number of Self-Attention blocks', type=int, default=0, required=True)

args = parser.parse_args()

assert args.model in ['resnet56', 'resnet44', 'resnet32', 'resnet20', 'sa', 'swga'], 'Unexpected model!'

############ varaiables

FINETUNING = False
log_interval = 100
batch_size = 128
test_batch_size = 128
epochs = 200
ealry_stopping_patience = 50
weight_decay = 1e-4
gamma_best = 0.
lambda_best = 0.
gamma_best_sigmoid = 0.
lambda_best_sigmoid = 0.

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
    save_path = os.path.join(save_path, now + '_' + args.model + "(" + str(args.n_blocks) + ")" + "_seed" + str(args.seed))   
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
        
    elif args.model == 'resnet44':
        model = models.resnet44()
        
    elif args.model == 'resnet32':
        model = models.resnet32()
        
    elif args.model == 'resnet20':
        model = models.resnet20()
    
    elif args.model == 'swga':
        if args.n_blocks < 27:
            model = models.self_attention_ResNet56(args.n_blocks, global_attribute=True)
        else:
            model = models.Self_Attention_full(global_attribute=True)
        # print(model)
    
    else:
        if args.n_blocks < 27:
            model = models.self_attention_ResNet56(args.n_blocks)
        else:
            model = models.Self_Attention_full()
        
    
    
        
    logger.debug(Fore.MAGENTA + Style.BRIGHT + '\n# Model: {}\
                                                \n# Initial Learning Rate: {}\
                                                \n# Seed: {}\
                                                \n# Weigth decay: {}'\
                                                .format(args.model + "(" + str(args.n_blocks) + ")", args.lr, args.seed, weight_decay) + Style.RESET_ALL)  
        
    if args.multi_gpus: # Using multi-gpu
        model = nn.DataParallel(model)
        print(Fore.RED + Style.BRIGHT + '\n# Multi Gpus Used!!' + Style.RESET_ALL)  
      
    model.cuda()
    
    summary(model, (3, 32, 32))    
    
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # print gamma value
    def get_gamma(model):
        
                
        for name, param in model.EBA.named_parameters():
            
            if '_gamma' in name:
                
                gamma_value = param.item()
                gamma_value_sigmoid = torch.sigmoid(torch.tensor(gamma_value))
                
                print(Fore.CYAN + Style.BRIGHT + '\nblock: {}\ngamma: {}\ngamma_sigmoid: {}'\
                                                    .format(name, gamma_value, gamma_value_sigmoid) + Style.RESET_ALL)
                                
            elif '_lambda' in name:
                
                lambda_value = param.item()
                lambda_value_sigmoid = torch.sigmoid(torch.tensor(lambda_value))
                
                print(Fore.CYAN + Style.BRIGHT + '\nblock: {}\nlambda: {}\nlambda_sigmoid: {}\n'\
                                            .format(name, lambda_value,
                                                    torch.sigmoid(torch.tensor(lambda_value))) + Style.RESET_ALL)
                             
        
        return gamma_value, lambda_value, gamma_value_sigmoid, lambda_value_sigmoid
    
    
    ############ trainers

    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001 )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)

    ############ training loop

    for epoch in tqdm(range(1, epochs+1)):
        # Train Mode
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()   # backpropagation 계산 전 opimizer 초기화
            #####################
            output = model(data)
            #####################    
            loss = F.cross_entropy(output, target)
            loss.backward()     # backpropagation 수행
            optimizer.step()    # weight update

            if batch_idx % log_interval == 0:
                logger.debug('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Test Mode
        # batch norm이나 droput 등을 train mode로 변환
        model.eval()
                
        test_loss = 0
        correct = 0
        with torch.no_grad():   # autograd engine, backpropagation이나 gradien 계산등을 꺼서 memory usage를 줄이고 속도 향상
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                #####################
                output = model(data)
                ##################### 
                test_loss += F.cross_entropy(output, target, reduction='sum').item()     # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)       # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuray = 100. * correct / len(test_loader.dataset)
        
        if args.model == 'swga':
            gamma_value, lambda_value, gamma_value_sigmoid, lambda_value_sigmoid = get_gamma(model)

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
                if 'swga' in args.model:
                    
                    gamma_best = gamma_value
                    gamma_best_sigmoid = gamma_value_sigmoid
                    lambda_best = lambda_value
                    lambda_best_sigmoid = lambda_value_sigmoid

        else:
            logger.debug(Fore.RED + Style.BRIGHT + 'best acc: {}'.format(early_stopping.best_value))
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

        # Tensorboard monitoring
        writer.add_scalar('Loss/train/', loss, epoch)    
        writer.add_scalar('Loss/test/', test_loss, epoch)
        writer.add_scalar('Accuracy/test/', test_accuray, epoch)
        writer.add_scalar('Learning Rate/', optimizer.param_groups[0]['lr'], epoch)
        if args.model == 'swga':
            writer.add_scalar('Gamma and Lambda/gamma/', gamma_value, epoch)
            writer.add_scalar('Gamma and Lambda/lambda/', lambda_value, epoch)
            writer.add_scalar('Gamma and Lambda/gamma_sigmoid/', gamma_value_sigmoid, epoch)
            writer.add_scalar('Gamma and Lambda/lambda_sigmoid/', lambda_value_sigmoid, epoch)

    logger.debug(Fore.RED + Style.BRIGHT + 'best acc: {}\
                                            \nmodel: {}\
                                            \nseed: {}\
                                            \nweight_decay: {}\
                                            \ntotal parameters: {}\
                                            \nbest gamma: {}\
                                            \nbest lambda: {}\
                                            \nbest gamma_sigmoid: {}\
                                            \nbest lambda_sigmoid: {}'\
                                            .format(early_stopping.best_value, args.model + "(" + str(args.n_blocks) + ")",\
                                                args.seed, weight_decay, params, gamma_best, 
                                                lambda_best, gamma_best_sigmoid, lambda_best_sigmoid))