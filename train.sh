#!/bin/zsh


for s in 3 4 5
do
	python main.py --depth 12 --channel 96 --heads 3 --gpu 0 --lr 0.0005 --model vit --tag vit-cifar10-12-96 --seed ${s} --dataset CIFAR10 --ls
#	python main.py --depth 12 --channel 144 --heads 3 --gpu 1 --lr 0.0005 --model vit --tag vit-cifar10-12-144 --seed ${s} --dataset CIFAR10 --ls
done
