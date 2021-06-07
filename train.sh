#!/bin/zsh

#for s in 1 2 3
#do
#	python main.py --depth 1 --gpu 1 --lr 0.003 --model g-pit --tag g-vit_LS_AA_pooling --seed ${s} --label_smoothing
	# python main.py --depth 2 --gpu 0 --lr 0.003 --model g-pit --tag g-vit_LS_AA_pooling --seed ${s} --label_smoothing
	# python main.py --depth 3 --gpu 0 --lr 0.003 --model g-pit --tag g-vit_LS_AA_pooling --seed ${s} --label_smoothing
#	python main.py --depth 4 --gpu 1 --lr 0.003 --model g-pit --tag g-vit_LS_AA_pooling --seed ${s} --label_smoothing
#done

for s in 1 2 3
do
	python main.py --depth 6 --channel 48 --heads 3 --gpu 1 --lr 0.003 --model g-vit --tag G-base --seed ${s} 
	# python main.py --depth 6 --channel 64 --heads 4 --gpu 0 --lr 0.003 --model g-vit --tag G-base --seed ${s} 
	# python main.py --depth 6 --channel 80 --heads 5 --gpu 0 --lr 0.003 --model g-vit --tag G-base --seed ${s} 
	python main.py --depth 6 --channel 96 --heads 6 --gpu 1 --lr 0.003 --model g-vit --tag G-base --seed ${s} 
done
 #python main.py --depth 6 --channel 80 --heads 5 --gpu 0 --lr 0.003 --model g-vit --tag g-vit_base_scale --seed 1
# python main.py --depth 6 --channel 96 --heads 6 --gpu 1 --lr 0.003 --model g-vit --tag g-vit_base_scale --seed 1
