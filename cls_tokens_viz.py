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
from visualization.ViT_T_M.model import Model
# from visualization.ViT_Masking.model import Model
from PIL import Image
from einops import rearrange

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    parser.add_argument('--tag', type=str, help='tag')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_path', default='./dataset/cifar100_img', type=str)

    return parser
# class GaussianSmoothing(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         channels (int, sequence): Number of channels of the input tensors. Output will
#             have this number of channels as well.
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super(GaussianSmoothing, self).__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         # The gaussian kernel is the product of the
#         # gaussian function of each dimension.
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#             ]
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
#                       torch.exp(-((mgrid - mean) / std) ** 2 / 2)

#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = kernel / torch.sum(kernel)

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
#         self.groups = channels

#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )

#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(input, weight=self.weight, groups=self.groups)

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
    model.load_state_dict(torch.load(os.path.join('./visualization/ViT_T_M', 'best.pth')))
    
    
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
        score = inference(transform(img).unsqueeze(dim=0), model, args) 

        cls_viz = rearrange(score, 'b c (h w) -> b c h w', h=int(math.sqrt(score.size(-1))))
        img = transforms.ToTensor()(img)
        img = rearrange(img, 'c h w -> h w c')
        img = img.detach().cpu().numpy()
        
        plt.rcParams["figure.figsize"] = (12,4)
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(img)
        ax1.set_title(f'Input Image', fontsize=12, pad=10)
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        
        # axs[0].plot(img.detach().cpu().numpy(),)    
        # axs[0].set_title(f'Input Image', fontsize=10)
        # axs[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax2 = plt.subplot(1, 3, 2)
        # sns.heatmap(cls_viz.squeeze(0).detach().cpu(), cmap='icefire', ax=ax2, vmin=0, vmax=1)
        # smoothing = GaussianSmoothing(1, 3, 1).cuda(args.gpu)
        cls_viz = transforms.Resize(32)(cls_viz)
        # cls_viz = F.pad(cls_viz, (1, 1, 1, 1), mode='reflect')
        # cls_viz = smoothing(cls_viz)
        tmp = cls_viz.squeeze()
        cls_viz = (tmp - torch.min(tmp)) / (torch.max(tmp)-torch.min(tmp))
        cls_viz = cls_viz.detach().cpu()
        ax2.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1)
        ax2.set_title(f'Class token`s Score map', fontsize=12, pad=10)
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(img)    
        ax3.imshow(cls_viz, cmap='rainbow', vmin=0, vmax=1, alpha=0.5)
        ax3.set_title(f'Blended image', fontsize=12, pad=10)
        ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                        
        plt.savefig(os.path.join(save_path, f'Class_Viz{i}.png'), format='png', dpi=400)
        
        # for i in range(len(scores)):
        #     if i==0 or (i+1) % 3 == 0:
        #         fig, axs = plt.subplots(3, 4, figsize=(28, 21))
        #         layer_viz = scores[i][0, :, 0, 1:]
        #         # layer_viz = scailing(layer_viz)
        #         layer_viz = rearrange(layer_viz, 'b (h w) -> b h w', h=int(math.sqrt(layer_viz.size(-1))))
        #         layer_viz = layer_viz.data
        #         for j, filter in enumerate(layer_viz):
        #             ax = axs.flat[j]
        #             sns.heatmap(filter.detach().cpu(), cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
        #             ax.set_title(f'{i+1}th depth / {j+1}th head', fontsize=5)
        #             ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                
        #         fig.savefig(os.path.join(save_path, f'scores_{i+1}.png'), format='png', dpi=1200)
        plt.cla()
        plt.clf()

def inference(img, model, args):
    print('inferencing')
    model.eval()
    with torch.no_grad():
    
        images = img.cuda(args.gpu, non_blocking=True)
        
        _ = model(images)
    
    

    cls = model.transformer.scores[-1][:, :, 0, 1:]
    mean_cls = cls.mean(dim=1, keepdim = True)

    return mean_cls              
            

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_clsviz_{args.tag}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args, save_path)
