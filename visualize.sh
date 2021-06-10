
for i in 10 11 12 13 14 15
do
	for j in 0.001 0.005 0.01 0.05 0.1 0.5
	do
		python visulization.py --model g-vit --depth 6 --heads 4 --channel 64 --tag test --n_neighbors ${i} --min_dist ${j}
	done
done

