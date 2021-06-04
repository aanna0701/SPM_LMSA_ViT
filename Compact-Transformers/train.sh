#!/bin/zsh

# for s in 1 2 3
# do
# 	# python main.py --depth 12 --channel 192 --heads 3 --gpu 1 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
# 	python main.py --depth 9 --channel 192 --heads 3 --gpu 0 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
# 	python main.py --depth 6 --channel 192 --heads 3 --gpu 0 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
# 	# python main.py --depth 3 --channel 192 --heads 3 --gpu 1 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
# done

for s in 2 3
do
	python main.py --depth 6 --channel 48 --heads 3 --gpu 1 --lr 0.003 --model vit --tag vit_base2 --seed ${s}
#	python main.py --depth 6 --channel 64 --heads 4 --gpu 0 --lr 0.003 --model vit --tag vit_base2 --seed ${s}
#	python main.py --depth 6 --channel 80 --heads 5 --gpu 0 --lr 0.003 --model vit --tag vit_base2 --seed ${s}
	python main.py --depth 6 --channel 96 --heads 6 --gpu 1 --lr 0.003 --model vit --tag vit_base2 --seed ${s}
done
python main.py --depth 6 --channel 80 --heads 5 --gpu 1 --lr 0.003 --model vit --tag vit_base2 --seed 1
python main.py --depth 6 --channel 96 --heads 6 --gpu 1 --lr 0.003 --model vit --tag vit_base2 --seed 1
