#!/bin/zsh


# for s in 3 4 5
# do
# 	python main.py --depth 12 --channel 96 --heads 3 --gpu 0 --lr 0.0005 --model vit --tag vit-cifar10-12-96-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu
# 	python main.py --depth 12 --channel 144 --heads 3 --gpu 1 --lr 0.0005 --model vit --tag vit-cifar10-12-144-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu
# done


for s in 3
do
	python main.py --depth 12 --heads 3 --channel 96 --gpu 1 --model pit --tag T-imgnet-12-96-aa-mu --seed ${s} --dataset T-IMNET --ls --aa --mu --down_conv
	python main.py --depth 12 --heads 3 --channel 144 --gpu 1 --model pit --tag T-imgnet-12-144-aa-mu --seed ${s} --dataset T-IMNET --ls --aa --mu --down_conv
#	python main.py --depth 12 --heads 3 --channel 96 --gpu 0 --model pit --tag cifar10-12-96-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu --down_conv
#	python main.py --depth 12 --heads 3 --channel 96 --gpu 0 --model pit --tag cifar10-12-144-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu --down_conv
	# python main.py --depth 12 --heads 3 --channel 96 --gpu 1 --model g-vit --tag cifar10-12-96-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu
	# python main.py  --depth 12 --heads 3 --channel 96 --channel 144 --gpu 1 --model g-vit --tag cifar10-12-144-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu 
done
