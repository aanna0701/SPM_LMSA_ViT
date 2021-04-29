import os
import logging as log
from models.model import *
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='get params')

parser.add_argument('--model', help='model', type=str, required=True)
parser.add_argument('--depth', help='depth', type=int, required=True)
parser.add_argument('--channel', help='channel', type=int, required=True)
parser.add_argument('--heads', help='heads', type=int, default=4)

args = parser.parse_args()

def get_params(model_name, depth, channel, heads):
        
    if model_name == 'ViT-Lite':
        model = ViT_Lite(depth, channel, heads = heads, dropout=False)
    elif model_name == 'G-ViT-Lite':
        model = ViT_Lite(depth, channel,GA=True, heads = heads, dropout=False)
    elif model_name == 'ViT-Lite-w_o-token':
        model = ViT_Lite(depth, channel,cls_token=False, heads = heads, dropout=False)
    elif model_name == 'G-ViT-Lite-w_o-token':
        model = ViT_Lite(depth, channel,GA= True,cls_token=False, heads = heads, dropout=False)
    
    save_path = os.path.join(os.getcwd(), 'params')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    
    log_dir = os.path.join(save_path, 'number of params_{}-{}-{}.txt'.format(model_name, depth, channel))
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'w')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)
    


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    logger.debug('****************************\
                \n# Model: {}-{}-{}\n{}\
                \n# # of Params: {}\n'
                 .format(model_name, model, depth, channel, params))
    

if __name__ == "__main__":
    get_params(args.model, args.depth, args.channel, args.heads)