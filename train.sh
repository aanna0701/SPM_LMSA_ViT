#!/bin/zsh

for s in  1 2 3
do
	for m in 3 8 
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.003 --model G-ViT-Lite --seed ${s} --depth ${m} --channel 96 --gpu 0 --heads 4
	done
done
