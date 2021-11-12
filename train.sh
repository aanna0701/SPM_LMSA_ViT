for s in 3 4 5
do 

#	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 0 --model pit --tag Residual_SPT_LSA_learnable_swin --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset 

	python main.py --heads 12 --channel 192 --depth 9 --lr 0.001 --gpu 1 --n_token 2 --type_trans trans_scale --model pit --tag test --init_type identity --padding border --scale 0.1 --is_trans_learn --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --data_path ../dataset  --n_trans 4 --lam 0
done
