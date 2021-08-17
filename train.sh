for s in 3 4 5
do 
	python main.py --depth 9 --heads 12 --channel 192 --lr 0.003 --gpu 1 --model g-vit --tag SPE-4-0.25 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
