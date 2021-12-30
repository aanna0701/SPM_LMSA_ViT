for s in 5 6 7
do 

	python main.py --lr 0.001 --gpu 0 --model swin --scale 0  --gam 0 --lam 0 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset 
	python main.py --heads 12 --channel 192 --depth 9 --lr 0.003 --gpu 0 --model vit --scale 0  --gam 0 --lam 0 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset


done
