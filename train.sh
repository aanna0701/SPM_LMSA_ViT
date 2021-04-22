#!/bin/zsh

for s in 1 2 3
do
	for m in 2 3 
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed ${s} --depth ${m} --channel 128
	done
done
