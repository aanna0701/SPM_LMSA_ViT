#!/bin/zsh

for s in 1 2 3 4 5
do
	for m in 1 2 3 4 5 6 7 8 9
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.1 --model sa --seed ${s} --n_blocks ${m}
		python train.py --dataset_dir /workspace/dataset --lr 0.1 --model swga --seed ${s} --n_blocks ${m}
	done
done
