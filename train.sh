#!/bin/zsh

for s in  1 2 3
do
	for m in 8 10
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.003 --model G-ViT-Lite --seed ${s} --depth ${m} --channel 72 --gpu 1 --heads 3 --tag CBAM_add_edge_aggregation
	done
done
