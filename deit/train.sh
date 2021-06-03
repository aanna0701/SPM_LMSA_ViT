#!/bin/zsh

for s in 1 2 3
do
	python main.py --data_path /workspace/dataset --gpu 1 --smoothing .1 --lr 0.0005 --model DeiT --seed ${s} --depth 3 --channel 192 --heads 3 --data_set CIFAR10 --mixup 0 --cutmix 0 --tag DeiT_base
	# python main.py --data_path /workspace/dataset --gpu 0 --smoothing .1 --lr 0.0005 --model DeiT --seed ${s} --depth 6 --channel 192 --heads 3 --data_set CIFAR10 --mixup 0 --cutmix 0 --tag DeiT_base
	# python main.py --data_path /workspace/dataset --gpu 0 --smoothing .1 --lr 0.0005 --model DeiT --seed ${s} --depth 9 --channel 192 --heads 3 --data_set CIFAR10 --mixup 0 --cutmix 0 --tag DeiT_base
	python main.py --data_path /workspace/dataset --gpu 1 --smoothing .1 --lr 0.0005 --model DeiT --seed ${s} --depth 12 --channel 192 --heads 3 --data_set CIFAR10 --mixup 0 --cutmix 0 --tag DeiT_base
	
done

