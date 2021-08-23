for s in 3 4 5
do 
	python main.py --heads 4 --lr 0.003 --gpu 0 --model g-vit2 --tag SPE --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
