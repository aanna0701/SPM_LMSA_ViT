#!/bin/zsh

# for s in 3 4 5
# do
# 	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model GiT --seed ${s} --depth 6 --channel 48 --heads 3 --tag GiT_ver1
# 	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model GiT --seed ${s} --depth 6 --channel 64 --heads 4 --tag GiT_ver1
# 	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model GiT --seed ${s} --depth 6 --channel 80 --heads 5 --tag GiT_ver1
# 	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model GiT --seed ${s} --depth 6 --channel 96 --heads 6 --tag GiT_ver1
	
# done

for s in  3 4 5
do
	# for d in 1 4
	# do
	# 	# python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model P-ViT-Node --seed ${s} --depth ${d} --channel 64 --heads 4 --tag P_ViT_Node_Baseline_quater --pp 4
	# done
	
	for d in 2 3
	do
		python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model P-ViT-Node --seed ${s} --depth ${d} --channel 64 --heads 4 --tag P_ViT_Node_Baseline_quater --pp 4
	done

	# for d in 1 4
	# do
	# 	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model P-ViT-Node --seed ${s} --depth ${d} --channel 64 --heads 4 --tag P_ViT_Node_Baseline_16 --r 4 --pp 16
	# done
	
	# for d in 2 3
	# do
	# 	python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model P-ViT-Node --seed ${s} --depth ${d} --channel 64 --heads 4 --tag P_ViT_Node_Baseline_half --r 4 --pp 16
	# done

done
