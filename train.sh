#!/bin/zsh

for s in  3 4 5
do
	for m in 5 10
	do
		python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model G-ViT-Lite --seed ${s} --depth ${m} --channel 72 --heads 3 --tag CBAM
	done
done
