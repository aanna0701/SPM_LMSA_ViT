for s in 3 4 5
do 

#	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 0 --model pit --tag Residual_SPT_LSA_learnable_swin --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset 

	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 0 --model vit --tag test --no_init --merging_size 4 --init_noise_trans 0 --init_noise_scale 0 --scale 0  --gam 1e-1 --lam 0 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset  --n_trans 4

done
