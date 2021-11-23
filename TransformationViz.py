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
import math


class Affine(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super().__init__()
        
        self.theta = None
        self.mode = padding_mode
        
    def forward(self, x, theta, init, scale=None):
        print('========')
        print(scale)
        print(theta[0])
        
        theta = theta.view(theta.size(0), -1)
        
        if scale is not None:
        
            theta = torch.mul(theta, scale)
        
        
        init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
        theta = torch.reshape(theta, (theta.size(0), 2, 3))    
        theta = theta + init 
        self.theta = theta    
        
        print(theta[0])
        
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid, padding_mode=self.mode)
    
   
# class Trans_scale(nn.Module):
#     def __init__(self, padding_mode='zeros'):
#         super().__init__()
#         self.mode = padding_mode
        
#     def forward(self, x, theta, init, scale=None):
        
#         print(x.shape)        
#         print(theta.shape)        
#         print(init.shape)        
#         print(scale.shape)        
        
        
#         init = torch.reshape(init.unsqueeze(0), (1, 2, 3)).expand(x.size(0), -1, -1) 
#         print('========')
#         print(theta[0])

#         # theta = torch.mul(theta, self.scale) + init
#         theta = theta + init if scale is None else torch.mul(theta, scale) + init
#         # theta = theta + init if scale is None else torch.mul(theta, scale) + torch.mul(init, (1-scale))
#         # theta = theta 
#         self.theta = theta
        
#         # theta = torch.reshape(theta, (theta.size(0), 2, 3))        
        
#         print(theta[0])
        
#         grid = F.affine_grid(theta, x.size())
        
#         return F.grid_sample(x, grid, padding_mode=self.mode)
     
    

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data', default='CIFAR100', type=str)

    return parser

def main(args, save_path):
 

    if args.data == 'CIFAR100':
        
        n_classes = 100
        img_size = 32
        patch_size = 4
        
        val_dataset = datasets.CIFAR100(
            root='./dataset', train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()]))
    
    elif args.data == 'T-IMNET':
        
        n_classes = 200
        img_size = 64
        patch_size = 8
        
        
        val_dataset = datasets.ImageFolder(
            root=os.path.join('../dataset', 'tiny_imagenet', 'val'), 
            transform=transforms.Compose([
            transforms.Resize(img_size), transforms.ToTensor()]))
    
        
    from visualization.Transformation.model import SwinTransformer
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
        
        
    model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, 
                            patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, 
                            num_heads=num_heads, num_classes=n_classes, is_base=False, 
                            num_trans=4, is_learn=True)
  
    
    # from visualization.ViT_SP_T_M.model import Model
    # model_vit_ours = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    checkpoint = torch.load(os.path.join('./visualization/Transformation', 'best.pth'))
    # model.load_state_dict(checkpoint)
    # const = None
    model.load_state_dict(checkpoint['model_state_dict'])
    # const = checkpoint['tr_constant']
    

    
    
    
    trans = Affine()
    
    for i, (images, _) in enumerate(val_dataset):
        
        fig_save_path = os.path.join(save_path, f'img{i}')
        os.makedirs(fig_save_path, exist_ok=True)

        plt.rcParams["figure.figsize"] = (5,5)
        img_raw = images.cuda(args.gpu, non_blocking=True)
        print(f'img {i}')       
        plt.imshow(img_raw.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(fig_save_path, f'{i}_input.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
        theta = model(img_raw.unsqueeze(0))
        init = model.patch_embed.patch_shifting.init
        scale = model.scale
        
        # for scale_list in scale:
        #     for scale in scale_list:
        #         print(scale)
        
        # print(theta)

        img_2 = transforms.Resize((img_raw.shape[1]//2, img_raw.shape[2]//2))(img_raw)
        img_4 = transforms.Resize((img_2.shape[1]//2, img_2.shape[2]//2))(img_2)
        # plt.imshow(img_2.permute(1, 2, 0).cpu().detach().numpy())
        # plt.savefig(os.path.join(fig_save_path, f'img_2.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
        
        # plt.imshow(img_4.permute(1, 2, 0).cpu().detach().numpy())
        # plt.savefig(os.path.join(fig_save_path, f'img_4.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
        
        
        
        # theta_list = torch.chunk(theta, 4, dim=1)
        for i, theta in enumerate(theta):
            
            if i == 0:
                for j, theta in enumerate(theta):
                    img_trans = trans(img_raw.unsqueeze(0), theta, init[j], scale[0][j])    
                    plt.imshow(img_trans.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
                    plt.savefig(os.path.join(fig_save_path, f'{j}_pe.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)

                     
            elif i == 1:   
                for j, theta in enumerate(theta):
                    img_trans = trans(img_2.unsqueeze(0), theta, init[j], scale[1][j])    
                    plt.imshow(img_trans.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
                    plt.savefig(os.path.join(fig_save_path, f'{j}_stage1.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
                
            elif i == 2:    
                for j, theta in enumerate(theta):
                    img_trans = trans(img_4.unsqueeze(0), theta, init[j], scale[2][j])    
                    plt.imshow(img_trans.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
                    plt.savefig(os.path.join(fig_save_path, f'{j}_stage2.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
            
        plt.clf()
       
if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_affine_{args.tag}_{args.data}')

    
    main(args, save_path)
