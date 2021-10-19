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


class Translation(nn.Module):
    def __init__(self):
        super(Translation, self).__init__()
        self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
        self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
        
      
        
    def forward(self, x,  theta):
        theta = torch.mul(theta, self.tmp1)
        theta = theta + self.tmp2.expand(x.size(0), 2, 3)
        
        # print(theta[0])
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        return F.grid_sample(x, grid, align_corners=True)
    

class Affine(nn.Module):
    def __init__(self):
        super().__init__()
              
        
    def forward(self, x,  theta):
                
        
        theta = torch.reshape(theta, (theta.size(0), 2, 3))
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        return F.grid_sample(x, grid, align_corners=True)

class Rigid(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp1 = torch.tensor([[0, 0, 1],[0, 0, 1]]).cuda(torch.cuda.current_device())
        self.tmp2 = torch.tensor([[1, 0, 0],[0, 1, 0]]).cuda(torch.cuda.current_device())
        self.tmp3 = torch.tensor([[0, -1, 0],[1, 0, 0]]).cuda(torch.cuda.current_device())
        
    def forward(self, x, theta):
        
    
        theta = theta 
        
        # print(theta[0])
        # print(epoch)
        # print(resolution)
        
        angle = theta[:, (0,)]
        trans = theta[:, 1:]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        mat_cos = torch.mul(cos, self.tmp2)
        mat_sin = torch.mul(sin, self.tmp3)
        mat_trans = torch.mul(trans, self.tmp1)
        
        theta = mat_cos + mat_sin + mat_trans
        
        grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)
    
    

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
        
        
    model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0, patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, is_base=False, num_trans=4)
  
    
    # from visualization.ViT_SP_T_M.model import Model
    # model_vit_ours = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model.load_state_dict(torch.load(os.path.join('./visualization/Transformation', 'best.pth')))
    
    
         
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    val_dataset = datasets.CIFAR100(
                root='./dataset', train=False, download=False, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()]))
    
    plt.rcParams["figure.figsize"] = (5,5)
    
    # trans = Affine()
    trans = Rigid()
    # trans = Translation()
    
    for i, (images, _) in enumerate(val_dataset):
        
        fig_save_path = os.path.join(save_path, f'img{i}')
        os.makedirs(fig_save_path, exist_ok=True)
    
        img_raw = images.cuda(args.gpu, non_blocking=True)
        print(f'img {i}')       
        plt.imshow(img_raw.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(fig_save_path, f'{i}_input.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
        _ = model(img_raw.unsqueeze(0))
        theta = model.theta
        theta_list = torch.chunk(theta, 4, dim=1)
        for j, theta in enumerate(theta_list):
            img_trans = trans(img_raw.unsqueeze(0), theta)    
            plt.imshow(img_trans.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
            plt.savefig(os.path.join(fig_save_path, f'{i}_{j}_trans.png'), format='png', dpi=400, bbox_inces='tight', pad_inches=0)
       


def inference(img, model, args):
    print('inferencing')
    model.eval()
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
    
    distributions =model.transformer.distributions
    
    dist_dict = {}
    none_dict = {}
    t_dict = {}
    t_m_dict = {}
    
    for i in range(distributions[0].size(1)):
        none_dict[i] = distributions[0][:, i]
        t_dict[i] = distributions[1][:, i]
        t_m_dict[i] = distributions[2][:, i]
        
    dist_dict['none'] = none_dict
    dist_dict['t'] = t_dict
    dist_dict['t_m'] = t_m_dict
    
    
    return dist_dict, distributions[0].size(2)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_Trans_{args.tag}')

    
    main(args, save_path)
