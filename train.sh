#!/bin/zsh

for s in 1 2 3
do
	for m in ViT-S
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.03 --model ${m} --seed ${s} --multi_gpus 
	done
done
