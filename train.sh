for s in 5 6 7
do 

	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 0 --model swin --margin 10 --pe_dim 48 --lam 0.01 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset
	python main.py --heads 12 --channel 192 --depth 9 --lr 0.003 --gpu 0 --model vit --margin 10 --pe_dim 96 --lam 0.01 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset

done
