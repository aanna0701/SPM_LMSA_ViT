
for s in 3 4 5
do
	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model vit --seed ${s} --depth 6 --channel 48 --heads 3 --tag ViT_base --dataset CIFAR10
	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model vit --seed ${s} --depth 6 --channel 64 --heads 4 --tag ViT_base --dataset CIFAR10
	# python train.py --dataset_dir /workspace/dataset --gpu 0 --lr 0.003 --model vit --seed ${s} --depth 6 --channel 80 --heads 5 --tag ViT_base --dataset CIFAR10
	python train.py --dataset_dir /workspace/dataset --gpu 1 --lr 0.003 --model vit --seed ${s} --depth 6 --channel 96 --heads 6 --tag ViT_base --dataset CIFAR10
done
