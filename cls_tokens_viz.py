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

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 't-imgnet'], type=str)

    return parser

def main(args, save_path):
  
    # if args.dataset == 'cifar100':        
    #     data_path = './dataset/cifar100_img'
    #     img_mean, img_std  = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    #     img_paths = glob.glob(os.path.join(data_path, '*.png'))  
    #     img_size = 32
    #     patch_size = 4
    #     num_classes = 100
        
    # elif args.dataset == 't-imgnet':
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
        
    print(Fore.GREEN+'*'*80)
    print(f"Creating model")    
    print('*'*80+Style.RESET_ALL)
    
    from visualization.ViT.model import Model
    model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model_vit.cuda(args.gpu)
    model_vit_ours.cuda(args.gpu)
    # model.load_state_dict(torch.load(os.path.join('./visualization/ViT_Masking', 'best.pth')))
    model_vit.load_state_dict(torch.load(os.path.join('./visualization/ViT', 'best.pth')))
    
    
         
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]


      
    random.seed(2)
    img_paths = random.sample(img_paths, 1000)    
    
    
    # img = Image.open(os.path.join('./visualization', 'input.png'))
    # img.save(os.path.join(save_path, 'input.png'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])

    
    for i, img_path in enumerate(img_paths):    
    
        img = Image.open(img_path)
        img = img.convert('RGB')
        score_vit = inference(transform(img).unsqueeze(dim=0), model_vit, args) 

        cls_viz_vit = rearrange(score_vit, 'b c (h w) -> b c h w', h=int(math.sqrt(score_vit.size(-1))))
        cls_viz_vit_ours = rearrange(score_vit_ours, 'b c (h w) -> b c h w', h=int(math.sqrt(score_vit_ours.size(-1))))
        img = transforms.ToTensor()(img)
        img = rearrange(img, 'c h w -> h w c')
        img = img.detach().cpu().numpy()
        
        plt.rcParams["figure.figsize"] = (25,5)
        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(img)
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        
        ax2 = plt.subplot(1, 5, 2)
        cls_viz = transforms.Resize(img_size)(cls_viz_vit)
        tmp = cls_viz.squeeze()
        cls_viz = (tmp - torch.min(tmp)) / (torch.max(tmp)-torch.min(tmp))
        cls_viz = cls_viz.detach().cpu()
        ax2.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1)
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax3 = plt.subplot(1, 5, 3)
        ax3.imshow(img)    
        ax3.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1, alpha=0.5)
        ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax4 = plt.subplot(1, 5, 4)
        cls_viz = transforms.Resize(img_size)(cls_viz_vit_ours)
        tmp = cls_viz.squeeze()
        cls_viz = (tmp - torch.min(tmp)) / (torch.max(tmp)-torch.min(tmp))
        cls_viz = cls_viz.detach().cpu()
        ax4.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1)
        ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax5 = plt.subplot(1, 5, 5)
        ax5.imshow(img)    
        ax5.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1, alpha=0.45)
        ax5.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                        
        plt.savefig(os.path.join(save_path, f'Class_Viz{i}.png'), format='png', dpi=400)
        

        plt.cla()
        plt.clf()

def inference(img, model, args):
    print('inferencing')
    model.eval()
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
    
    
    cls = model.transformer.score[:, :, 0, 1:]
    mean_cls = cls.mean(dim=1, keepdim = True)

    return mean_cls              
            

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_clsviz_{args.tag}')
    # save_path = os.path.join('./visualization', f'results_clsviz_{args.tag}_train')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
