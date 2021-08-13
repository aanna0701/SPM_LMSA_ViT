for s in 3 4 5
do 
	python main.py --depth 9 --heads 6 --channel 384 --lr 0.001 --gpu 1 --model cvt --tag base --seed ${s} --dataset T-IMNET --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
