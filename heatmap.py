import glob
import os
import argparse

from numpy.lib.index_tricks import AxisConcatenator
import cv2
import shutil
import math
import numpy as np

def init_parser():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--path', required=True, type=str, help='dataset path')

    return parser



def main(args, save_dir):
    paths = glob.glob(os.path.join(args.path, '*'))


    for path in paths:
        results_dict = {}
        filter_names_set = set()
        save_dir = os.path.join(path, 'heatmap')
        os.makedirs(save_dir, exist_ok=True)
        imgs = glob.glob(os.path.join(path, '*'))
        imgs.sort()
        for img in imgs:
            filter_name = img.split('/')[-1]
            
            if 'png' in filter_name:
                filter_names_set.add(filter_name.split('_')[0])
            for n in filter_names_set:
                if not 'png' in n:
                    if not n in results_dict:
                        results_dict[n] = []
                    os.makedirs(os.path.join(save_dir, n), exist_ok=True)
                    
            if 'png' in filter_name:
                idx = filter_name.split('_')[0]
                if not 'png' in idx:
                    save_f = img.split('_')[0]
                    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    # heatmap_img = cv2.applyColorMap(im, cv2.COLORMAP_JET)
                    results_dict[idx].append(im)
                    
                else:
                    shutil.copyfile(img, os.path.join(save_dir, idx))

        for key in results_dict:
            results_dict[key] = combine_images(results_dict[key]).astype(np.uint8)
            # print(np.shape(results_dict[key]))
            heatmap_img = cv2.applyColorMap(results_dict[key], cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, key, 'heatmap.png'), heatmap_img)
            
        # print(type(results_dict['f8']))
    
def combine_images(imgs):
    y = 0
    v = []
    
    # length = len(imgs) // 12 # 16
    # for i in range(12):
    #     v_tmp = imgs[i*length:i*length + length]
    #     v_tmp = np.concatenate((v_tmp), axis=1)
    #     v.append(v_tmp)
    # out = np.concatenate((v), axis=0)
    
    # print(np.shape(imgs))
    
    return np.mean(imgs, axis=0)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
        
    REP = 3
    SAVE_DIR = args.path
    
    main(args, SAVE_DIR)