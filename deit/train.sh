#!/bin/zsh

for s in 1 2 3
do
	# python main.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 48 --heads 3
	python main.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 64 --heads 4
	python main.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 80 --heads 5 
	# python main.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model DeiT --seed ${s} --depth 6 --channel 96 --heads 6 
	
done

