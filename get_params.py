import os
import logging as log

import numpy as np
import argparse
import models.create_model as m


parser = argparse.ArgumentParser(description='get params')

parser.add_argument('--model', help='model', type=str, required=True)
parser.add_argument('--depth', help='depth', type=int, required=True)
parser.add_argument('--channel', help='channel', type=int, required=True)
parser.add_argument('--heads', help='heads', type=int, default=4)
parser.add_argument('--tag', help='tag', required=True)

args = parser.parse_args()

def get_params(model_name, depth, channel, heads):
        
    # model load

    if args.model == 'ViT':
        model = m.make_ViT(args.depth, args.channel, heads = args.heads, dropout=False)
    elif args.model == 'GiT':
        model = m.ViT_Lite(args.depth, args.channel,GA=True, heads = args.heads, dropout=False)
    elif args.model == 'P-ViT-Max':
        model = m.P_ViT_max(args.depth, args.channel, heads = args.heads, dropout=False)
    elif args.model == 'P-ViT-Conv':
        model = m.P_ViT_conv(args.depth, args.channel, heads = args.heads, dropout=False)
    elif args.model == 'P-ViT-Node':
        model = m.P_ViT_node(args.depth, args.channel, heads = args.heads, dropout=False)
    elif args.model == 'P-GiT-Max':
        model = m.P_GiT_max(args.depth, args.channel, GA=True,heads = args.heads, dropout=False)
    elif args.model == 'P-GiT-Node':
        model = m.P_GiT_node(args.depth, args.channel, GA=True,heads = args.heads, dropout=False)
    
    save_path = os.path.join(os.getcwd(), 'params')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    
    log_dir = os.path.join(save_path, 'number of params_{}-{}-{}-{}-{}.txt'.format(model_name, depth, heads, channel, args.tag))
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