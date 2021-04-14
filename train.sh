#!/bin/zsh

for s in 1 2 3
do
	for m in G-ViT-Ti ViT-Ti
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.003 --model ${m} --seed ${s}
	done
done
