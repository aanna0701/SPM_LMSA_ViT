#!/bin/zsh

for s in 1 2
do
	for m in 4 6
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.003 --model ViT-Lite --seed ${s} --depth ${m} --channel 128 --gpu 0 --heads 4
	done
done
