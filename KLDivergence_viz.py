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
colors = sns.color_palette('Paired',8)

# plt.rcParams["xtick.labelsize"] = 'xx-large'
# plt.rcParams["ytick.labelsize"] = 'xx-large'

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--data_path', default='./dataset/', type=str)
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', 'T-IMNET', 'SVHN'], type=str)

    return parser



def main(args, save_path):
    
    i=0

    
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
        patch_size = 4
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
        patch_size = 4
        
    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        print('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
        patch_size = 8
   
    '''
        Model 
    '''    
    
    


        
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    '''
        Data Loader
    '''
    if args.dataset == 'CIFAR10':
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':

        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'T-IMNET':
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor(), *normalize]))

    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=100)
  
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    
    avg = {}  
    
    

    '''
        ViT
    '''
        
    from visualization.ViT.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT', 'best.pth'))))
    model.cuda(args.gpu)

    values = inference(val_loader, model)

    name = 'ViT'
    plot_values(values, name, i, colors[i*2+1])
    i+=1
    avg[name] = compute_avg(values)


    '''
        ViT-T
    '''
        
    from visualization.ViT_T.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T', 'best.pth'))))
    model.cuda(args.gpu)
    
    values = inference(val_loader, model)

    name = 'ViT-T'
    plot_values(values, name, i, colors[i*2+1])
    i+=1
    avg[name] = compute_avg(values)

    '''
        ViT-M
    '''
        
    from visualization.ViT_M.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_M', 'best.pth'))))
    model.cuda(args.gpu)
    
    values = inference(val_loader, model)

    name = 'ViT-M'
    plot_values(values, name, i, colors[i*2+1])
    i+=1   
    avg[name] = compute_avg(values)
        
    '''
        ViT-T-M
    '''
        
    from visualization.ViT_T_M.model import Model
    model = Model(img_size=img_size, patch_size=patch_size, num_classes=n_classes)
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T_M', 'best.pth'))))
    model.cuda(args.gpu)
    
    values = inference(val_loader, model)

    name = 'ViT-LSA'
    plot_values(values, name, i, colors[i*2+1])
    i+=1
    avg[name] = compute_avg(values)
    
    plt.legend(loc='upper center', fontsize='xx-large', ncol=4)
    plt.ylim([0, 1.9])    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.locator_params(axis="y", nbins=5)
    plt.grid(axis='y')
    # plt.xlabel('Depth', fontsize=20)
    
    plt.savefig(os.path.join(save_path, 'KLDivergence.png'))
    plt.clf()

    avgList = avg.items()
    # avgList = sorted(avgList) 
    x, y = zip(*avgList)
    
    c = []
    for i in range(4):
        c.append(colors[2*i+1])
        
    plt.bar(x, y, color=c, width=0.5, zorder=3)
    for index, value in enumerate(y):
        plt.text(index, value+0.05, f'{value:.4f}', fontsize='xx-large', ha='center')
    plt.ylim([0, 1.2])
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.locator_params(axis="y", nbins=5)
    plt.grid(axis='y')
    
    plt.savefig(os.path.join(save_path, 'AVG_KLDivergence.png'))
    



def inference(val_loader, model):
    print('inferencing')
    with torch.no_grad():
        for _, (images, _) in enumerate(val_loader):
            images = images.cuda(args.gpu)
            _ = model(images)
            
    return model.transformer.KLD  

def plot_values(values, name, i, colors):
    valueList = values.items()
    valueList = sorted(valueList) 
    x, y = zip(*valueList) 
    x = [int(x)+1 for x in x]
    
    plt.plot(x, y, label=name, marker=Markers[i], color=colors)
  
    
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
    save_path = os.path.join('./visualization', f'smoothing_idx_{args.dataset}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
