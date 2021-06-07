#!/bin/zsh

for s in 1 2 3
do
	python main.py --data_path /workspace/dataset/imnet --gpu 0 --lr 0.0005 --model DeiT --seed ${s} --depth 12 --channel 192 --heads 3 --data_set IMNET --tag G_DeiT_Ti --data_set IMNET
	python main.py --data_path /workspace/dataset/imnet --gpu 0 --lr 0.0005 --model DeiT --seed ${s} --depth 12 --channel 384 --heads 6 --data_set IMNET --tag G_DeiT_B --data_set IMNET
done

