#!/bin/zsh

for s in 1 2 3
do
	python main.py --data_path /workspace/dataset/IMNET --gpu 0 --lr 0.0005 --model DeiT --seed ${s} --depth 3 --channel 192 --heads 3 --data_set IMNET --tag G_DeiT_Ti
	python main.py --data_path /workspace/dataset/IMNET --gpu 0 --lr 0.0005 --model DeiT --seed ${s} --depth 6 --channel 192 --heads 3 --data_set IMNET --tag G_DeiT_B
	
done

