#!/bin/zsh

for s in  3 4 5
do
	for m in 5 8
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.003 --model G-ViT-Lite --seed ${s} --depth ${m} --channel 72 --gpu 1 --heads 3 --tag node_maxpool_edge_MHSA_out
	done
done
