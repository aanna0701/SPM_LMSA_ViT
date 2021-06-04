#!/bin/zsh

for s in 1 2 3
do
	# python main.py --depth 12 --channel 192 --heads 3 --gpu 1 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
	python main.py --depth 9 --channel 192 --heads 3 --gpu 0 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
	python main.py --depth 6 --channel 192 --heads 3 --gpu 0 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
	# python main.py --depth 3 --channel 192 --heads 3 --gpu 1 --lr 0.0005 --model DeiT --tag deit_test --seed ${s}
done

