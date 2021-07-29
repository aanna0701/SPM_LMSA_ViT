import umap
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from colorama import Fore, Style
import os
from visualization.ViT_T_M.model import Model

def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path', default='/dataset', type=str, help='dataset path')
   
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100','IMNET'], type=str, help='Image Net dataset path')

    parser.add_argument('--n_neighbors', default=5, type=int, help='10 to 15')
    parser.add_argument('--min_dist', default=0.3, type=float, help='0.001 to 0.5')

    parser.add_argument('--tag', type=str, help='tag')

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')

    return parser



def main(args, save_path):
    '''
        Dataset
    '''
    global n_neighbors
    global min_dist

    n_neighbors = [3, 4]
    min_dist = [0.2, 0.3]
    
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32
        
    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        print('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32
   
    '''
        Model 
    '''    

    model = Model()


        
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

    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False)
    
    model.load_state_dict((torch.load(os.path.join('./visualization/ViT_T_M', 'best.pth'))))
    
    token, cls = inference(val_loader, model, args)
    fig, axs = plt.subplots(len(n_neighbors), len(min_dist))
    for i in range(len(n_neighbors)):
        args.n_neighbors = n_neighbors[i]
        for j in range(len(min_dist)):
            args.min_dist = min_dist[j]
            embedding = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist).fit_transform(token)

            visualize(embedding, cls, f'n_neighbors: {args.n_neighbors} min_dist: {args.min_dist}', axs[i, j])
    # fig.colorbar(ticks=range(10))
    
    # fig.savefig(os.path.join(save_path, f'result.eps'), format='eps', dpi=1200)
    fig.savefig(os.path.join(save_path, f'result.png'), format='png', dpi=1200)

def inference(val_loader, model, args):
    print('inferencing')
    model.eval()
    embedding_list = []
    with torch.no_grad():
        for _, (images, target) in enumerate(val_loader):
            
            _ = model(images)
            concat = torch.cat([model.final_cls_token, target.unsqueeze(-1)], dim=-1) # (B, C+1)
            embedding_list.append(concat)
    embedding = torch.cat(embedding_list, dim=0)    # (N, C+1)
            
    
    return embedding[:, :-1].numpy(), embedding[:, -1].numpy()

def visualize(embeddings, cls, title, axs=False):
    print('visualizing')
       
    
    # vis_x = embeddings[:, 0]
    # vis_y = embeddings[:, 1]
    # plt.scatter(vis_x, vis_y, c=cls, cmap=plt.cm.get_cmap('spectral', 10), marker='.')
    # plt.colorbar(ticks=range(10))
    # plt.title(title, fontsize=15)
    # plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # plt.savefig(os.path.join(save_path, f'result_{args.n_neighbors}_{args.min_dist}.eps'), format='eps', dpi=1200)
    # plt.savefig(os.path.join(save_path, f'result_{args.n_neighbors}_{args.min_dist}.png'), format='png', dpi=1200)
       
    
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    axs.scatter(vis_x, vis_y, c=cls, cmap=plt.cm.get_cmap('Paired', 10), marker='.', s=0.1, )
    axs.set_title(title, fontsize=5)
    axs.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # plt.savefig(os.path.join(save_path, f'result_{args.n_neighbors}_{args.min_dist}.eps'), format='eps', dpi=1200)
    # plt.savefig(os.path.join(save_path, f'result_{args.n_neighbors}_{args.min_dist}.png'), format='png', dpi=1200)



if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    # global model_name
    save_path = os.path.join('./visualization', f'results_umap_{args.tag}')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(args,save_path)