for s in 3 4 5
do 
	python main.py --depth 9 --heads 6 --channel 384 --lr 0.001 --gpu 0 --model cait --tag SP_T_M --seed ${s} --dataset CIFAR10 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
