#!/bin/zsh

ARRAY_MODEL=("resnet56", "nlb2")
ARRAY_SEED=(1, 2, 3)

for m in "resnet56" "nlb2"
do
	for s in 1 2 3
	do
		python train.py --lr 0.1 --model ${m} --seed ${s}
	done
done
