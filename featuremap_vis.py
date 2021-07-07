import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
import os
from visualization.ViT_Masking.model import Model
import glob
from random import sample
from PIL import Image

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
    model.load_state_dict(torch.load(os.path.join('./visualization/ViT_Masking', 'best.pth')))
    
    
    img_mean, img_std  = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)     
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    img_paths = glob.glob(os.path.join(args.data_path, '*'))    
    img_path = sample(img_paths, 1)
    
    # img = Image.open(img_path[0])
    img = Image.open(os.path.join('./visualization', 'input.png'))
    img.save(os.path.join(save_path, 'input.png'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])
    
    img_tensor = transform(img).unsqueeze(dim=0)
    
    scores = inference(img_tensor, model, args)
    

    
    
    for i in range(len(scores)):
        fig, axs = plt.subplots(3, 4, figsize=(28, 21))
        layer_viz = scores[i][0, :, :, :]
        layer_viz = scailing(layer_viz)
        layer_viz = layer_viz.data
        for j, filter in enumerate(layer_viz):
            ax = axs.flat[j]
            sns.heatmap(filter.detach().cpu(), cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
            ax.set_title(f'{i+1}th depth / {j+1}th head', fontsize=5)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
          
        fig.savefig(os.path.join(save_path, f'scores_{i+1}.png'), format='png', dpi=1200)
        plt.cla()
        plt.clf()

def scailing(x):
    x_min, _ = x.min(dim=-1, keepdim = True)
    x_max, _ = x.max(dim=-1, keepdim = True)
    out = torch.div(x - x_min, x_max - x_min)
    return out

def inference(img, model, args):
    print('inferencing')
    model.eval()
    embedding_list = []
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
        
    return model.transformer.scores    
            
            

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_{args.tag}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
