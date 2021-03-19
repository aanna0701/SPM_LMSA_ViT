#!/bin/zsh

for m in "resnet56" "nlb1" "nlb2" "nlb3" "nlb4"
do
	for s in 1 2 3
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.1 --model ${m} --seed ${s}
	done
done
