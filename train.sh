for s in 7 8 9
do 
	python main.py --depth 9 --heads 12 --channel 192 --lr 0.001 --gpu 1 --model cvt --tag base --seed ${s} --dataset T-IMNET --ls --aa --mu --sd 0.1 --ra 3 --cm --re 0.25
done
