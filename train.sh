

# for s in 3 4 5
# do
# 	python main.py --depth 12 --channel 96 --heads 3 --gpu 0 --lr 0.0005 --model vit --tag vit-cifar10-12-96-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu
# 	python main.py --depth 12 --channel 144 --heads 3 --gpu 1 --lr 0.0005 --model vit --tag vit-cifar10-12-144-aa-mu --seed ${s} --dataset CIFAR10 --ls --aa --mu
# done


for s in 3 4 5
do
	python main.py --depth 9 --heads 12 --channel 192 --lr 0.001 --gpu 1 --model g-vit --tag git_new_ver2 --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25 --ver 2
#	python main.py --depth 12 --heads 6 --channel 192 --lr 0.001 --gpu 0 --model vit --tag h-12_c-192_fixed --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --depth 12 --heads 12 --channel 192 --lr 0.001 --gpu 0 --model vit --tag h-12_c-192_fixed --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --depth 9 --heads 3 --channel 192 --lr 0.001 --gpu 0 --model g-vit --tag heads_2-6-4_wide --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done

for s in 3
do
#	python main.py --gpu 1 --model vgg16 --tag aa-mu-cm-sd-ra-re --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --gpu 1 --model res56 --tag aa-mu-cm-sd-ra-re --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --gpu 1 --model resXt --tag aa-mu-cm-sd-ra-re --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --gpu 1 --model mobile2 --tag aa-mu-cm-sd-ra-re --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
#	python main.py --gpu 1 --model dense121 --tag aa-mu-cm-sd-ra-re --seed ${s} --dataset CIFAR100 --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
