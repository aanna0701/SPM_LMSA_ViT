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
from torchvision.utils import save_image
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
    data_path = '../dataset/tiny_imagenet/val'
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
    
    # """ Model """
    # from visualization.ViT.model import Model
    # model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)    
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)       
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/ViT', 'best.pth')))
    
    # from visualization.LS_ViT.model import Model
    # model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/LS_ViT', 'best.pth')))
    
    
    # from visualization.PiT.model import Model
    # model_vit = Model(img_size=img_size, patch_size=patch_size//2, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/PiT', 'best.pth')))
    
    # from visualization.LS_PiT.model import Model
    # model_vit = Model(img_size=img_size, patch_size=patch_size//2, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/LS_PiT', 'best.pth')))
    
    
    # from visualization.T2T.model import Model
    # model_vit = Model(img_size=img_size, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/T2T', 'best.pth')))
    
    # from visualization.LS_T2T.model import Model
    # model_vit = Model(img_size=img_size, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/LS_T2T', 'best.pth')))
    
    
    # from visualization.Swin.model import Model
    # model_vit = Model(img_size=img_size, patch_size= patch_size//2, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/Swin', 'best.pth')))
    
    """ Model """
    from visualization.CaiT.model import Model
    model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)    
    torch.cuda.set_device(args.gpu)
    model_vit.cuda(args.gpu)       
    model_vit.load_state_dict(torch.load(os.path.join('./visualization/CaiT', 'best.pth')))
    
    # from visualization.LS_CaiT.model import Model
    # model_vit = Model(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
    # torch.cuda.set_device(args.gpu)
    # model_vit.cuda(args.gpu)   
    # model_vit.load_state_dict(torch.load(os.path.join('./visualization/LS_CaiT', 'best.pth')))
    
         
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

      
    random.seed(1)
    img_paths = random.sample(img_paths, 100)    
    
    
    # img = Image.open(os.path.join('./visualization', 'input.png'))
    # img.save(os.path.join(save_path, 'input.png'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])

    fig = plt.figure()
    
    for i, img_path in enumerate(img_paths):    
    
        img = Image.open(img_path)
        img = img.convert('RGB')
        score_vit = inference(transform(img).unsqueeze(dim=0), model_vit, args) 

        cls_viz_vit = rearrange(score_vit, 'b c (h w) -> b c h w', h=int(math.sqrt(score_vit.size(-1))))
        img = transforms.ToTensor()(img)
        img = rearrange(img, 'c h w -> h w c')
        img = img.detach().cpu().numpy()
        
        plt.rcParams["figure.figsize"] = (5,5)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(os.path.join(save_path, f'{i}_input.png'), format='png', dpi=400, bbox_inches='tight', pad_inches = 0)

        
        cls_viz = transforms.Resize(img_size)(cls_viz_vit)
        tmp = cls_viz.squeeze()
        cls_viz = (tmp - torch.min(tmp)) / (torch.max(tmp)-torch.min(tmp))
        cls_viz = cls_viz.detach().cpu()
  
        plt.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1, alpha=0.5)
        
        plt.savefig(os.path.join(save_path, f'{i}_heatmap.png'), format='png', dpi=400, bbox_inches='tight', pad_inches = 0)
        
        plt.clf()

        
       

def inference(img, model, args):
    print('inferencing')
    model.eval()
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
    
    """ ViT """
    # cls = model.transformer.score[:, :, 0, 1:]
    
    """ PiT """
    # cls = model.layers[4].score[:, :, 0, 1:]
    
    """ T2T """
    # cls = model.blocks[11].score[:, :, 0, 1:]
    
    """ CaiT """
    cls = model.cls_transformer.score[:, :, 0, 1:]
    # print(cls.shape)
    
    """ Swin """
    # print(model.layers[2].blocks[3].score)
    # cls = model.layers[2].blocks[3].score
    # cls = rearrange(cls, 'b c h w -> b c (h w)')
    
    
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
