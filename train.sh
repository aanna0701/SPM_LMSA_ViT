for s in 5
do 
	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 1 --model swin --tag SPM_wo_pool --seed ${s} --dataset CIFAR10 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 
done
