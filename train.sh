#!/bin/zsh

# for s in  3 4 5
# do
# 	#python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model ViT --seed ${s} --depth 6 --channel 48 --heads 3 --tag ViT_Baseline
# 	#python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model ViT --seed ${s} --depth 6 --channel 64 --heads 4 --tag ViT_Baseline
# 	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model ViT --seed ${s} --depth 9 --channel 64 --heads 4 --tag ViT_Baseline
# 	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model ViT --seed ${s} --depth 12 --channel 64 --heads 4 --tag ViT_Baseline
	
# done

for s in  3 4 5
do
	for d in 2 3
	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model P-ViT-Conv-Pooling --seed ${s} --depth ${d} --channel 64 --heads 4 --tag P_ViT_Conv_Baseline
	
done
