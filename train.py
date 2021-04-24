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
# from adamp import AdamP
from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

import models.model as m

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"



# constants

init(autoreset=True)
use_cuda = True
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
now = datetime.now().strftime('%Y-%m-%d-%H_%M')

# args

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--dataset_dir', help='path of input images',
                    default='/media/CVIP/Hyundai2020/dataset/training/0809')
parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
parser.add_argument('--model', help='model', required=True)
parser.add_argument('--gpu', help='gpu number to use', default='multi')
parser.add_argument('--seed', help='seed', type=int, required=True)
parser.add_argument('--depth', help='depth', type=int, required=True)
parser.add_argument('--channel', help='channel', type=int, required=True)
parser.add_argument('--weights', help='weights path', default=False)
# parser.add_argument('--n_blocks', help='number of Self-Attention blocks',
#                     type=int, default=0, required=True)

args = parser.parse_args()

assert args.model in ['ViT-Lite', 'PiT-Lite','G-ViT-Lite', 'G-PiT-Lite', 'ViT-Lite-w_o-token', 'G-ViT-Lite-w_o-token'], 'Unexpected model!'

# gpus
# GPU 할당 변경하기

if not args.gpu == 'multi':
    GPU_NUM = int(args.gpu) # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

# random seed

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# varaiables

FINETUNING = False
log_interval = 100
batch_size = 128
test_batch_size = 128
epoch_init = 1
epochs = 200
ealry_stopping_patience = 50
weight_decay = 0.03
gamma_dict_list_best = []
lambda_dict_list_best = []
best_train_loss = 100000
best_train_accuracy = 0


def model_eval(data_loader):
    loss = 0
    accuracy = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            #####################
            output = model(data)
            #####################
            # sum up batch loss
            loss += F.cross_entropy(output,
                                    target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return loss, accuracy


if __name__ == "__main__":

    # save path

    save_path = os.path.join(os.getcwd(), "save")
    # save_path = os.path.join(save_path, now + '_' + args.model +
    #                          "(" + str(args.n_blocks) + ")" + "_seed" + str(args.seed))
    save_path = os.path.join(save_path, now + '_' + args.model + '_{}_{}'.format(args.depth, args.channel) +
                             "_seed" + str(args.seed))
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    # logger

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

    # training objects

    writer = SummaryWriter()
    early_stopping = EarlyStopping(
        patience=ealry_stopping_patience, verbose=1, mode='max')

    # data loaders

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, 4),
                             transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                         ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                         ])),
        batch_size=batch_size, shuffle=True)

    # model load

    if args.model == 'ViT-Lite':
        model = m.ViT_Lite(args.depth, args.channel)
    elif args.model == 'G-ViT-Lite':
        model = m.ViT_Lite(args.depth, args.channel,GA=True)
    elif args.model == 'ViT-Lite-w_o-token':
        model = m.ViT_Lite(args.depth, args.channel,cls_token=False)
    elif args.model == 'G-ViT-Lite-w_o-token':
        model = m.ViT_Lite(args.depth, args.channel,GA= True,cls_token=False)
        
    # trainers

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 200, max_lr=args.lr, min_lr=0.00005, warmup_steps=10)


    logger.debug(Fore.MAGENTA + Style.BRIGHT + '\n# Model: {}\
                                                \n# Initial Learning Rate: {}\
                                                \n# Seed: {}\
                                                \n# depth: {}\
                                                \n# channel: {}\
                                                \n# Weigth decay: {}'
                 .format(args.model, args.lr, args.seed, args.depth, args.channel, weight_decay) + Style.RESET_ALL)



    if args.gpu=='multi':  # Using multi-gpu
        model = nn.DataParallel(model)
        print(Fore.RED + Style.BRIGHT + '\n# Multi Gpus Used!!' + Style.RESET_ALL)
    else:
        print(Fore.RED + Style.BRIGHT + '\n# Using Gpus {}'.format(torch.cuda.get_device_name(GPU_NUM)))
        
    if args.weights:
        checkpoint = torch.load(os.path.join(args.weights, best.pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']
        loss = checkpoint['loss']


    model.cuda()

    summary(model, (3, 32, 32))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # print gamma value
    def get_gamma(model):

        gamma_dict_list = []
        labmda_dict_list = []

        for name, param in model.named_parameters():

            if '_gamma' in name:

                gamma_dict = {}

                gamma_value = param.item()
                gamma_value_sigmoid = torch.sigmoid(torch.tensor(gamma_value))

                gamma_dict['name'] = name
                gamma_dict['gamma'] = gamma_value
                gamma_dict['gamma_sigmoid'] = gamma_value_sigmoid

                print(Fore.CYAN + Style.BRIGHT + '='*25 + Style.RESET_ALL)
                print(Fore.CYAN + Style.BRIGHT + '\nblock: {}\ngamma: {}\ngamma_sigmoid: {}'
                      .format(name, gamma_value, gamma_value_sigmoid) + Style.RESET_ALL)

                gamma_dict_list.append(gamma_dict)

            elif '_lambda' in name:

                lambda_dict = {}

                lambda_value = param.item()
                lambda_value_sigmoid = torch.sigmoid(
                    torch.tensor(lambda_value))

                lambda_dict['name'] = name
                lambda_dict['lambda'] = lambda_value
                lambda_dict['lambda_sigmoid'] = lambda_value_sigmoid

                print(Fore.CYAN + Style.BRIGHT + '\nblock: {}\nlambda: {}\nlambda_sigmoid: {}\n'
                      .format(name, lambda_value,
                              torch.sigmoid(torch.tensor(lambda_value))) + Style.RESET_ALL)

                labmda_dict_list.append(lambda_dict)

        return gamma_dict_list, labmda_dict_list

    

    # training loop

    for epoch in tqdm(range(epoch_init, epochs+1)):
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

        train_loss, train_accuracy = model_eval(train_loader)
        test_loss, test_accuracy = model_eval(test_loader)

        logger.debug(Fore.BLUE + Style.BRIGHT +
                     '\ntrain_loss {}\ntrain_accuracy {}\n\ntest_loss {}\ntest_accuracy {}\n'.format(train_loss, train_accuracy, test_loss, test_accuracy))

        if best_train_accuracy < train_accuracy:
            best_train_accuracy = train_accuracy
        if best_train_loss > train_loss:
            best_train_loss = train_loss

        # early stopping
        early_stop = early_stopping.validate(test_accuracy)

        if not early_stop:
            if early_stopping.best_value <= test_accuracy:
                logger.debug(Fore.GREEN + Style.BRIGHT +
                             '\nbest model updates!!!!!\n')
                # model.state_dict(): 딕셔너리 형태로 모델의 Layer와 Weight가 저장되어있음.
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_path+'/best.pt')
                gamma_dict_list_best, lambda_dict_list_best = get_gamma(
                    model)

        else:
            logger.debug(Fore.RED + Style.BRIGHT +
                         '\nbest test acc: {}\nbest_train_loss: {}\nbest_train_acc: {}'.format(early_stopping.best_value, best_train_loss, best_train_loss))
            # model.state_dict(): 딕셔너리 형태로 모델의 Layer와 Weight가 저장되어있음.
            torch.save({
                'epoch': epoch + 1,
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
        writer.add_scalar('Accuracy/test/', test_accuracy, epoch)
        writer.add_scalar('Learning Rate/',
                          optimizer.param_groups[0]['lr'], epoch)

    logger.debug(Fore.RED + Style.BRIGHT + '\nbest val acc: {}\nbest train acc: {}\nbest train loss: {}\
        \nmodel: {}-{}-{}\nseed: {}\nweight_decay: {}\ntotal parameters: {}\nbest gamma: {}\nbest lambda: {}'
                 .format(early_stopping.best_value, best_train_accuracy, best_train_loss, args.model, args.depth, args.channel,
                         args.seed, weight_decay, params, gamma_dict_list_best,
                         lambda_dict_list_best))
