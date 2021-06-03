#!/bin/zsh

for s in 1 2 3
do
#	python main.py --data_path /workspace/dataset --gpu 1 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 48 --heads 3 --data_set CIFAR10 --mixup 0 --cutmix 0
	python main.py --data_path /workspace/dataset --gpu 0 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 64 --heads 4 --data_set CIFAR10 --mixup 0 --cutmix 0
	python main.py --data_path /workspace/dataset --gpu 0 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 80 --heads 5 --data_set CIFAR10 --mixup 0 --cutmix 0
#	python main.py --data_path /workspace/dataset --gpu 1 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 96 --heads 6 --data_set CIFAR10 --mixup 0 --cutmix 0
	
done

