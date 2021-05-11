#!/bin/zsh

for s in  3 4 5
do
	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model ViT-Lite --seed ${s} --depth 6 --channel 48 --heads 3 --tag ViT_Baseline
	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model ViT-Lite --seed ${s} --depth 6 --channel 80 --heads 4 --tag ViT_Baseline
	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model ViT-Lite --seed ${s} --depth 6 --channel 64 --heads 4 --tag ViT_Baseline
	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model ViT-Lite --seed ${s} --depth 6 --channel 96 --heads 4 --tag ViT_Baseline
	
done
