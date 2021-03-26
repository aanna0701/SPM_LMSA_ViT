#!/bin/zsh

for s in 1 2 3 4 5
do
	for m in 12 15 18 21 24 27
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.05 --model sa --seed ${s} --n_blocks ${m}
		python train.py --dataset_dir /workspace/dataset --lr 0.05 --model swga --seed ${s} --n_blocks ${m}
	done
done
