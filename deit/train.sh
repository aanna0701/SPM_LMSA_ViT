#!/bin/zsh

for s in 1 2 3
do
	#python main.py --data_path /workspace/dataset/imnet --gpu 0 --lr 0.0005 --model g-vit --seed ${s} --depth 12 --channel 192 --heads 3 --tag g-deit-Ti
	python main.py --data_path /workspace/dataset/imnet --gpu 0 --lr 0.0005 --model g-vit --seed ${s} --depth 12 --channel 384 --heads 6 --tag g-deit-B 
done

