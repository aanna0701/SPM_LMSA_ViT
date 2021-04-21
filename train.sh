#!/bin/zsh

for s in 1 2 3
do
	for m in ViT-Lite
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ${m} --seed ${s} --gpu 1 --depth 4 --channel 56
	done
done
