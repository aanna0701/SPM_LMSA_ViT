import argparse
# import torch
# import torch.optim
# import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
# from colorama import Fore, Style
import os
# pandas 사용하기
import numpy as np # numpy 도 함께 import
import pandas as pd


Markers = ['o', 'X', 'D', 's']
F_size = 8
plt.rcParams["figure.figsize"] = (4,3.5)
plt.rcParams["axes.axisbelow"] = True
colors = sns.color_palette('Paired', 20)
# colors = ['r', 'g', 'b', 'k', 'y']

# plt.rcParams["xtick.labelsize"] = 'xx-large'
# plt.rcParams["ytick.labelsize"] = 'xx-large'

# Data Frame 정의하기
# 이전에 DataFrame에 들어갈 데이터를 정의해주어야 하는데,
# 이는 python의 dictionary 또는 numpy의 array로 정의할 수 있다.



def main(save_path):
    
    data_ViT = {'name': ['ViT', 'T2T', 'PiT', 'Swin', 'CaiT'],
            'acc': [57.07, 60.57, 60.25, 60.87, 64.37],
            'params': [2.8, 6.7, 7.06, 7.13, 9.1]}
    data_ours = {'name': ['SL-ViT', 'SL-T2T', 'SL-PiT', 'SL-Swin', 'SL-CaiT'],
            'acc': [61.07, 61.83, 62.91, 64.95, 67.18],
            'params': [2.9, 7.1, 8.7, 10.2, 9.2]}
    
    data_ViT = {'name': ['ViT', 'SL-ViT'],
            'acc': [57.07,61.07],
            'params': [2.8, 2.9]}
    data_PiT = {'name': ['PiT', 'SL-PiT'],
            'acc': [60.25,62.91],
            'params': [7.06, 8.7]}
    data_T2T = {'name': ['T2T', 'SL-T2T'],
            'acc': [60.57,61.83],
            'params': [6.7, 7.1]}
    data_Swin = {'name': ['Swin', 'SL-Swin'],
            'acc': [60.87,64.95],
            'params': [7.13, 10.2]}
    data_CaiT = {'name': ['CaiT', 'SL-CaiT'],
            'acc': [64.37, 67.18],
            'params': [9.1, 9.2]}
    
    # df_base = pd.DataFrame(data_base)
    # df_ours = pd.DataFrame(data_ours)


    # scatter_plot=df_base.plot.scatter(x='params',y='acc')
    # scatter_plot.plot()
    
    i = 1
    plt.rc('font', family='serif')
    
    plt.grid()
    scatter(data_ViT, i)
    
    # plt.text(data_ViT['params'][1]+0.2, data_ViT['acc'][1]+0.4, data_ViT['name'][1], ha='center', fontsize=F_size+2)
    plt.text(data_ViT['params'][1]+1.3, data_ViT['acc'][1], data_ViT['name'][1], ha='center', fontsize=F_size+2)
    
    i += 2   
    scatter(data_PiT, i)
    
    plt.text(data_PiT['params'][1]+1.2, data_PiT['acc'][1]-0.2, data_PiT['name'][1], ha='center', fontsize=F_size+2)
    
    i += 2 
    scatter(data_T2T, i)
    
    plt.text(data_T2T['params'][1], data_T2T['acc'][1]+0.5, data_T2T['name'][1], ha='center', fontsize=F_size+2)
    
    # # plt.savefig(os.path.join(save_path, 'No_pool.png'))
    # # plt.clf()
    i += 2
    scatter(data_Swin, i)
    
    plt.text(data_Swin['params'][1]+0.45, data_Swin['acc'][1]+0.5, data_Swin['name'][1], ha='center', fontsize=F_size+2)

    i += 2 
    scatter(data_CaiT, i)
    
    plt.text(data_CaiT['params'][1], data_CaiT['acc'][1]+0.5, data_CaiT['name'][1], ha='center', fontsize=F_size+2)
    
    plt.legend(loc=2, fontsize=F_size+1)
    plt.ylim([56.5, 68.5])
    plt.xlim([2, 12])
    # plt.rcParams.update({'font.size': 22})
    plt.xlabel('The Number of Parameters (M)', fontsize=F_size+1)
    plt.ylabel('Tiny-ImageNet Top-1 Accuracy (%)', fontsize=F_size+1)
    plt.xticks(fontsize=F_size)
    plt.yticks(fontsize=F_size)
    plt.savefig(os.path.join(save_path, 'pool.png'))
        
    
    
def scatter(data, i):
    # base plot
    x = data['params']
    y = data['acc']
    plt.scatter(x, y, color=colors[i], s=20)
    plt.plot(x, y, color=colors[i], label=data['name'][0], linestyle='dashed')
    # ours plot
    x = data['params'][1]
    y = data['acc'][1]
    plt.scatter(x, y, color=colors[i], marker='*',s=80)
    plt.plot(x, y)

    
if __name__ == '__main__':
    # global model_name
    save_path = os.path.join('./visualization', f'main_fig')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    
    main(save_path)
