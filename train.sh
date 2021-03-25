#!/bin/zsh

for m in 9 12 18
do
	for s in 1 2 3
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.1 --model sa --seed ${s} --n_blocks ${m}
		python train.py --dataset_dir /workspace/dataset --lr 0.1 --model swga --seed ${s} --n_blocks ${m}
	done
done
