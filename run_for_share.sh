#!/bin/bash

for i in 1 2
do
	for j in 1 2 3
	do
    	python main.py --tr 0.6 --path ./datasets/ --dropout 0.1 --weight_decay 1e-4 --alpha 1e-4 --lr 1e-2 --epochs 200 --extrastr 1 --dataset BlogCatalog \
	 --normy 1 --nin $i --nout $j --hidden 100 --clip 100.
	done
	# for j in 1e-6
	# do
	# 	python main.py --tr $i --dropout 1.0 --weight_decay 1e-6 --alpha $j --lr 1e-2 --epochs 50 --extrastr 1
	# done
done
