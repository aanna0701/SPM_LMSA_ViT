#!/bin/zsh

for s in 2 3
do
	for m in 128 256
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w/o-token --seed ${s} --gpu 1 --depth 4 --channel ${m}
	done
done
