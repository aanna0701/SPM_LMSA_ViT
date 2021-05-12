#!/bin/zsh

for d in  1 2 3 4 5

do
    python get_params.py --model P-GiT-Conv-Pooling --depth ${d} --channel 48 --heads 3 --tag ViT
done

