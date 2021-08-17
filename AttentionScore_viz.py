import argparse
import torch
from torch import nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
import math
import numbers
from colorama import Fore, Style
import os
# from visualization.ViT_Masking.model import Model
from PIL import Image
from einops import rearrange
import glob
import random
import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=5)
import seaborn as sns
colors = sns.color_palette('Paired',16)

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='t-imgnet', choices=['cifar100', 't-imgnet'], type=str)

    return parser

def main(args, save_path):
 
        
    if args.dataset == 't-imgnet':
        data_path = './dataset/tiny_imagenet/val'
        # data_path = './dataset/tiny_imagenet/train'
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        folder_paths = glob.glob(os.path.join(data_path, '*'))  
        img_paths = []
        for path in folder_paths:
            img_paths = img_paths + glob.glob(os.path.join(path, '*'))
        img_size = 64
        patch_size = 8
        num_classes = 200

    
    from visualization.ViT_T_M.model import Model
    model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    # from visualization.ViT_SP_T_M.model import Model
    # model_vit_ours = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model_vit.cuda(args.gpu)
    # model_vit_ours.cuda(args.gpu)
    # model.load_state_dict(torch.load(os.path.join('./visualization/ViT_Masking', 'best.pth')))
    model_vit.load_state_dict(torch.load(os.path.join('./visualization/ViT_T_M', 'best.pth')))
    # model_vit_ours.load_state_dict(torch.load(os.path.join('./visualization/ViT_SP_T_M', 'best.pth')))
    
    
         
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    random.seed(2)
    img_paths = random.sample(img_paths, 100)    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])

    for i, img_path in enumerate(img_paths):
        
        fig_save_path = os.path.join(save_path, f'img{i}')
        os.makedirs(fig_save_path, exist_ok=True)
    
        img = Image.open(img_path)
        img = img.convert('RGB')
        dist_dict, x_size = inference(transform(img).unsqueeze(dim=0), model_vit, args) 
        # x_ticks = list(range(1, x_size+1))
        
        sample_x = 8
        x_ticks = list(range(1, sample_x+1))
        
        sample = list(range(0, x_size-1, x_size // sample_x))
        
        
        plt.rcParams["figure.figsize"] = (4,2)
        # plt.rcParams['axes.axisbelow'] = True
        
        color_idx = 0
        max_flag = 0
        
        
        for key in dist_dict:            
            dists = dist_dict[key]
            type_name = key
            color = colors[4*color_idx+1]            
            color_idx += 1
            if not key == 'none':
                max_flag = 1
            
            for key in dists:
                # dist = F.softmax(dists[sample])
                dist = dists[key].squeeze().detach().cpu().numpy()[sample]
                dist = softmax(dist)
                fig_name = type_name + f'_{key}'
                
                
                plt.plot(x_ticks, dist, color=color)
                # if max_flag:
                #     f = lambda i: dist[i]
                #     max_idx = max(range(len(dist)), key=f)
                #     plt.text(max_idx, dist[max_idx]+0.001, f'{dist[max_idx]:.4f}', fontsize='xx-large', ha='center')
                plt.fill_between(x_ticks, dist, alpha=0.6, color=color)
                plt.ylim([0, 0.25])
                plt.xlim([1, sample_x])    
                plt.locator_params(axis="y", nbins=5)
        
                # plt.grid(axis='y')
                
                plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                plt.savefig(os.path.join(fig_save_path, fig_name+'.png'), format='png', dpi=400)
                plt.clf()


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
    save_path = os.path.join('./visualization', f'results_DistViz_{args.tag}')

    
    main(args, save_path)
