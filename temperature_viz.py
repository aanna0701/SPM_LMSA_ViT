import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import os

Markers = ['o', 'X', 'D', 's']
plt.rcParams["figure.figsize"] = (10,5)
colors = sns.color_palette('Paired',4)

# plt.rcParams["xtick.labelsize"] = 'xx-large'
# plt.rcParams["ytick.labelsize"] = 'xx-large'

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_path', default='./dataset/', type=str)
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', 'T-IMNET', 'SVHN'], type=str)

    return parser



def main(args, save_path):
    i = 1

    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_size = 32
        patch_size = 4
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_size = 32
        patch_size = 4
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        print('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_size = 64
        patch_size = 8


    # '''
    #     CIFAR10
    # '''
    # n_classes = 10
    # img_size = 32
    # patch_size = 4
        
    # from visualization.ViT_T.model import Model
    # model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    # model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T/CIFAR10', 'best.pth'))))


    # name = 'CIFAR10'
    # plot_values(model, name, colors[i])
    # i+=2
    # plt.hlines(4, 1, 9, color='red')
    # plt.legend(loc='upper center', fontsize='xx-large', ncol=4)
    # plt.ylim([0, 6])    
    # plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.locator_params(axis="y", nbins=5)
    # plt.grid(axis='y')
    
    # # plt.savefig(os.path.join(save_path, f'{name}.png'))
    # # plt.clf()

    '''
        CIFAR100
    '''
    n_classes = 100
    img_size = 32
    patch_size = 4
        
    from visualization.ViT_T.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T/CIFAR100', 'best.pth'))))


    name = 'CIFAR100'
    plot_values(model, name, 0, colors[i])
    i+=2
    # plt.hlines(4, 1, 9, color='red')
    plt.legend(loc='upper center', fontsize='xx-large', ncol=4)

    '''
        T-IMNET
    '''
    n_classes = 200
    img_size = 64
    patch_size = 8
        
    from visualization.ViT_T.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T/T-IMNET', 'best.pth'))))


    name = 'T-IMNET'
    plot_values(model, name, 1, colors[i])
    i+=2
    plt.hlines(4, 1, 9, color='red', linestyles='dashed')
    plt.legend(loc='upper center', fontsize='xx-large', ncol=4)
    plt.ylim([0, 6])    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.locator_params(axis="y", nbins=6)
    plt.grid(axis='y')    
    plt.rc('font', family='serif')

    # '''
    #     SVHN
    # '''
    # n_classes = 100
    # img_size = 32
    # patch_size = 4
        
    # from visualization.ViT_T.model import Model
    # model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    # model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T/SVHN', 'best.pth'))))


    # name = 'SVHN'
    # plot_values(model, name, colors[i])
    # i+=2
    # plt.hlines(4, 1, 9, color='red')
    # plt.legend(loc='upper center', fontsize='xx-large', ncol=4)
    # plt.ylim([0, 6])    
    # plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.locator_params(axis="y", nbins=5)
    # plt.grid(axis='y')
    
    plt.savefig(os.path.join(save_path, f'ViT_results.png'))
    plt.clf()
       

def extract_temperature(model):
    temperature_max = []
    temperature_mean = []
    temperature_min = []
    for name, param in model.named_parameters():
        if 'scale' in name:
            inverse = torch.div(1, param.data)
            t_mean =inverse.mean()
            t_max = inverse.max() - t_mean
            t_min = t_mean - inverse.min()
            temperature_max.append(t_max.item())
            temperature_mean.append(t_mean.item())
            temperature_min.append(t_min.item())
      
    return temperature_max, temperature_mean, temperature_min

def plot_values(model, name, i, colors):
    # valueList = values.items()
    # valueList = sorted(valueList) 
    # x, y = zip(*valueList) 
    
    t_max, t_mean, t_min = extract_temperature(model)
    x, y = range(len(t_mean)), t_mean
    x = [x+1 for x in x]
    yerr = (t_min, t_max)    
    # plt.errorbar(x, y, yerr=yerr, label=name, color=colors, marker=Markers[i])
    plt.plot(x, y, label=name, color=colors, marker=Markers[i])
  
    
def compute_avg(values):
    
    tmp = 0
    
    for key in values:
        tmp += values[key]
    
    tmp /= len(values)
        
    return tmp
    
if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'temperature_{args.dataset}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
