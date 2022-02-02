import argparse
import torch
from torch import nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
import math
from colorama import Fore, Style
import os
# from visualization.RES110.model import resnet110 as Model
from visualization.ViT.model import Model
# from visualization.ViT_Masking.model import Model
from PIL import Image
from einops import rearrange

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_path', default='./dataset/cifar100_img', type=str)

    return parser

def main(args, save_path):
  
  
    print(Fore.GREEN+'*'*80)
    print(f"Creating model")    
    print('*'*80+Style.RESET_ALL)
    
    model = Model()
    
    '''
    GPU
    '''
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # model.load_state_dict(torch.load(os.path.join('./visualization/ViT_Masking', 'best.pth')))
    model.load_state_dict(torch.load(os.path.join('./visualization/ViT', 'best.pth')))
    # model.load_state_dict(torch.load(os.path.join('./visualization/RES110', 'best.pth')))
    
    
    img_mean, img_std  = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)     
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    import glob
    import random
    img_paths = glob.glob(os.path.join(args.data_path, '*'))    
    random.seed(1)
    img_paths = random.sample(img_paths, 100)    
    
    
    # img = Image.open(os.path.join('./visualization', 'input.png'))
    # img.save(os.path.join(save_path, 'input.png'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])
    
    # img_tensor = transform(img).unsqueeze(dim=0)
    
    for i, img_path in enumerate(img_paths):    
        
        
    
        img = Image.open(img_path)
        img.save(os.path.join(save_path, f'{i}_input.png'))
        features = inference(transform(img).unsqueeze(dim=0), model, args) 
        
        img = transforms.ToTensor()(img)
        img = rearrange(img, 'c h w -> h w c')
        img = img.detach().cpu().numpy()
        
        for key in features:
            
            feauturemap = features[key]
            feauturemap = scailing(feauturemap)
            feauturemap = rearrange(feauturemap, 'b (h w) -> b h w', h=int(math.sqrt(feauturemap.size(-1))))
            feauturemap = transforms.Resize(32)(feauturemap)
            feauturemap = feauturemap.squeeze(dim=0)
            feauturemap = (feauturemap - torch.min(feauturemap)) / (torch.max(feauturemap)-torch.min(feauturemap))
            feauturemap = feauturemap.detach().cpu()
          
            
            if key == 'patch_embedding':
                plt.rcParams["figure.figsize"] = (8,4)
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(img)
                ax1.set_title(f'Input Image', fontsize=8, pad=5)
                ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
          
                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(feauturemap, cmap='coolwarm', vmin=0, vmax=1)
                ax2.set_title(f'Patch embedding', fontsize=8, pad=5)
                ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
                plt.savefig(os.path.join(save_path, f'{i}_Patch Embedding.png'), format='png', dpi=400)
                plt.clf()
                
            title = f'Layer{key}\'s Featuremap' if not key == 'patch_embedding' else f'Patch embedding'
            
            plt.rcParams["figure.figsize"] = (20,4)
                       
            ax = plt.subplot(2, 5, int(key)+2) if not key == 'patch_embedding' else plt.subplot(2, 5, 1)
            ax.imshow(feauturemap, cmap='coolwarm', vmin=0, vmax=1)
            ax.set_title(title, fontsize=8, pad=5)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
            plt.savefig(os.path.join(save_path, f'{i}_Featuremaps.png'), format='png', dpi=400)

def scailing(x):
    x_min, _ = x.min(dim=-1, keepdim = True)
    x_max, _ = x.max(dim=-1, keepdim = True)
    out = torch.div(x - x_min, x_max - x_min)
    return out


def inference(img, model, args):
    print('inferencing')
    model.eval()
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
    
    

    features = model.featuremaps
    
    return features
            

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_featuremaps_{args.tag}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
