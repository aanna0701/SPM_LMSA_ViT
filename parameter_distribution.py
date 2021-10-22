import argparse
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import os
import torch.nn as nn
# from visualization.ViT_Masking.model import Model
from PIL import Image
import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=5)
import seaborn as sns
colors = sns.color_palette('Paired',16)
    
    

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)

    return parser

def main(args, save_path):
 
        
    n_classes = 100
    img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
    img_size = 32
    patch_size = 4
    in_channels = 3

    
    from visualization.PCA.model import SwinTransformer
    if img_size > 64:
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        mlp_ratio = 4
        window_size = 7
        patch_size = 4
    else:
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size //= 2
        
        
    model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, is_base=False, num_trans=4)
  
    
    # from visualization.ViT_SP_T_M.model import Model
    # model_vit_ours = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model.load_state_dict(torch.load(os.path.join('./visualization/PCA', 'best.pth')))
    
    
         
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    
         

    val_dataset = datasets.CIFAR100(
                root='./dataset', train=False, download=False, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                *normalize]))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, pin_memory=True)
    

    # trans = Affine()
    # trans = Translation()
    
    theta_list = list()
    
    for i, (images, _) in enumerate(val_loader):
    
        img_raw = images.cuda(args.gpu, non_blocking=True)
        
        theta = model(img_raw) * patch_size
        
        
        theta_list.append(theta)

    theta = torch.cat(theta_list, dim=0)
    theta = torch.chunk(theta, 4, dim=-1)
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rc('font', family='serif')
    plt.ylim([-0.4, 0.4])
    plt.xlim([-0.4, 0.4])
    
    
    # pca_theta = torch.pca_lowrank(theta[0], niter=10)
    
    # print(pca_theta[0].shape)
    
    labels = ['T1', 'T2', 'T3', 'T4']
    
    
    
    for i, x in enumerate(theta):
        
        
    
        x = x.cpu().detach().numpy()
        print(labels[i])
    
        for j in range(x.shape[0]):
                      
            if j == 0:
                plt.scatter(x[j][0], x[j][1], color=colors[2*i+1], label=labels[i], s=3, alpha=0.5)
                
            elif j < 1000:
                plt.scatter(x[j][0], x[j][1], color=colors[2*i+1], s=3, alpha=0.5)
                
            else:
                break
            
            print(j)
    
    plt.legend(loc='upper center', fontsize='large', ncol=4, bbox_to_anchor=(0.5, 1.12), fancybox=True)
    plt.title(f'{args.tag}', pad=43, fontsize='large')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'result.png'), format='png', dpi=400)         

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./PCA', f'results_{args.tag}')
    os.makedirs(save_path, exist_ok=True)
    
    main(args, save_path)
