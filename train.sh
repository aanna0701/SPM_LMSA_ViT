#!/bin/zsh

for s in 1 2 3
do
	python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed ${s} --depth 18 --channel 96 --gpu 0 --heads 4	
	for m in 8 10 18
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed ${s} --depth ${m} --channel 64 --gpu 0 --heads 4
	done
done
