#!/bin/zsh

python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed 1 --gpu 1 --depth 4 --channel 128
python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed e --gpu 1 --depth 4 --channel 128

for s in 2 3
do
	for m in 256
	do
		python train.py --dataset_dir /workspace/dataset --lr 0.0005 --model ViT-Lite-w_o-token --seed ${s} --gpu 1 --depth 4 --channel ${m}
	done
done
