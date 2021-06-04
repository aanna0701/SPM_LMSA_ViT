import os
import sys
from colorama import init, Fore, Style, Back
import argparse
import numpy as np
from training_functions import EarlyStopping
import logging as log
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
# from adamp import AdamP
from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from deit.autoaugment import CIFAR10Policy
import models.create_model as m
from time import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models.losses import LabelSmoothingCrossEntropy


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"



# constants

use_cuda = True
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# args

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--dataset_dir', help='path of input images',
                    default=None)
parser.add_argument('--lr', help='Learning Rate', default=0.01, type=float)
parser.add_argument('--model', help='model', required=True)
parser.add_argument('--gpu', help='gpu number to use', default='multi')
parser.add_argument('--seed', help='seed', type=int, required=True)
parser.add_argument('--depth', help='depth', type=int, required=True)
parser.add_argument('--channel', help='channel', type=int, required=True)
parser.add_argument('--heads', help='heads', type=int, default=4)
parser.add_argument('--tag', help='description of this training', required=True)
parser.add_argument('--weights', help='weights path', default=False)
parser.add_argument('--dataset', help='dataset', type=str, default='CIFAR10')
parser.add_argument('--able_aug', action='store_true')
parser.add_argument('--label_smoothing', action='store_true')
# parser.add_argument('--n_blocks', help='number of Self-Attention blocks',
#                     type=int, default=0, required=True)

args = parser.parse_args()

assert args.model in ['vit', 'g-vit', 'deit', 'g-deit'], 'Unexpected model!'
assert args.dataset in ['CIFAR10', 'CIFAR100', 'IMNET'], 'Unexpected dataset!'
# gpus
# GPU 할당 변경하기

if not args.gpu == 'multi':
    GPU_NUM = int(args.gpu) # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

'''
random seed
'''

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

'''
varaiables
'''

FINETUNING = False
log_interval = 10
batch_size = 128
test_batch_size = 128
epochs = 300
ealry_stopping_patience = 50
weight_decay = 0.03
gamma_dict_list_best = []
lambda_dict_list_best = []
best_train_loss = 100000
best_train_accuracy = 0
global n_classes
if args.dataset == 'CIFAR10':
    n_classes = 10
    img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
elif args.dataset == 'CIFAR100':
    n_classes = 100
    img_mean, img_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
elif args.dataset == 'IMNET':
    n_classes = 1000
    img_mean, img_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


save_path = os.path.join(os.getcwd(), "save")
save_path = os.path.join(save_path, args.model + f'-{args.depth}-{args.heads}-{args.channel}' + "_seed" + str(args.seed) + f"-{args.tag}")
os.makedirs(save_path, exist_ok=True)

'''    
logger
'''

log_dir = os.path.join(save_path, 'log.txt')
logger = log.getLogger(__name__)
formatter = log.Formatter('%(message)s')
streamHandler = log.StreamHandler()
fileHandler = log.FileHandler(log_dir, 'w')
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)
logger.setLevel(level=log.DEBUG)


####################################################################################

def model_train(model, data_loader, optimizer, criterion, epoch):
    model.train()
    train_loss, train_accuracy = 0, 0
    n = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()   # backpropagation 계산 전 opimizer 초기화
        #####################
        output = model(data)
        #####################
        n += data.size(0)
        loss = criterion(output, target)
        loss.backward()     # backpropagation 수행
        optimizer.step()    # weight update
        train_loss += loss.item() * data.size(0)
        train_accuracy += accuracy(output, target) * data.size(0)

        if batch_idx % log_interval == 0:
            avg_loss, avg_acc = train_loss/n, train_accuracy/n
            logger.debug(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tTrain Loss: {avg_loss:.6f}\tTrain Acc: {avg_acc:6.2f}\tLR: {get_lr(optimizer):.6f}')
    
    return train_loss/n, train_accuracy/n


def model_eval(model, data_loader, criterion, epoch):
    eval_loss = 0
    eval_accuracy = 0
    model.eval()
    n = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            #####################
            output = model(data)
            #####################
            n += data.size(0)
            # sum up batch loss
            eval_loss += criterion(output,target).item() * data.size(0)
            eval_accuracy += accuracy(output, target) * data.size(0)
            
            if batch_idx % log_interval == 0:
                avg_loss, avg_acc = eval_loss/n, eval_accuracy/n
                logger.debug(f'Eval Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tEval Loss: {avg_loss:.6f}\tEval Acc: {avg_acc:6.2f}')
    
    eval_loss /= n
    eval_accuracy /= n
    return eval_loss, eval_accuracy


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return float(res[0])


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def main(args, save_path):
    # model load
    if args.model == 'deit':
        model = m.make_ViT(args.depth, args.channel, heads = args.heads, num_classes=n_classes)
    elif args.model == 'g-deit':
        model = m.make_ViT(args.depth, args.channel,GA=True, heads = args.heads, num_classes=n_classes)
    elif args.model == 'vit':
        model = m.make_ViT(args.depth, args.channel,GA=False, heads = args.heads, num_classes=n_classes)
        args.able_aug = True    
    elif args.model == 'g-vit':
        model = m.make_ViT(args.depth, args.channel,GA=True, heads = args.heads, num_classes=n_classes)
        args.able_aug = True
    
    
    print(Fore.RED  + '*'*20)
    logger.debug(f'Model: {args.model}\nInitial Learning Rate: {args.lr}\nSeed: {args.seed}\ndepth: {args.depth}\nheads: {args.heads}\nchannel: {args.channel}\nWeigth decay: {weight_decay}')      
    
    if args.gpu=='multi':  # Using multi-gpu
        model = nn.DataParallel(model)
        print('Multi Gpus Used!!')
    else:
        print(f'Using Gpus {torch.cuda.get_device_name(GPU_NUM)}')      
   
    # data loaders
    augmentations = []
    
    if args.able_aug:
        print('Auto augmentation used')
        augmentations += [CIFAR10Policy()]  
            
           
    augmentations += [transforms.RandomHorizontalFlip(),                             
                        transforms.RandomCrop(32, padding=4),                             
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=True, download=True,
                            transform=transforms.Compose(augmentations)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_dir, train=False, download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])), batch_size=batch_size, shuffle=True)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 300, max_lr=args.lr, min_lr=1e-6, warmup_steps=5)
    
    if args.label_smoothing:
        print('Label smoothing used')
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    print('*'*20 + Style.RESET_ALL)
    model.cuda(device)      
    criterion.cuda(device)  
    
    epoch_init = 0
    if args.weights:
        checkpoint = torch.load(os.path.join(args.weights, 'best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch'] + 1
        train_loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    summary(model, (3, 32, 32))    
    best_acc = 0  
    print()
    print("Beginning training")
    print()
    time_begin = time()
    
    for epoch in range(epoch_init, epochs):
        train_loss, train_acc = model_train(model, train_loader, optimizer, criterion, epoch)
        eval_loss, eval_acc = model_eval(model, test_loader, criterion, epoch)        
        logger.debug(f'[Train]\t\t Loss: {train_loss:.6f}\t\t Acc: {train_acc:6.2f}')
        logger.debug(f'[Eval]\t\t Loss: {eval_loss:.6f}\t\t Acc: {eval_acc:6.2f}')
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            print(Fore.BLUE  + '*'*80)
            logger.debug(f'[Best model update] Best accuracy: {best_acc:6.2f}')
            print('*'*80 + Style.RESET_ALL)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'scheduler_state_dict': scheduler.state_dict()
                }, save_path+'/best.pt')            
        
        # Learning Rate Schedule
        scheduler.step()
        
    total_mins = (time() - time_begin) / 60
    print(Fore.YELLOW + '*'*20)
    logger.debug(f'Training is done \t \t Best Eval accuracy: {best_acc:.2f} \t \t # of parameters: {n_parameters} \t \t Training time: {total_mins:.2f} minutes')
    print('*'*20)
    
    
    torch.save(model.state_dict(), save_path+'/final.pt')

if __name__ == "__main__":

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)      
        
    main(args, save_path)

