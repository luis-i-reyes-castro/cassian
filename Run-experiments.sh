#!/bin/bash

STORE=103
NUM_RUNS=4
EPOCHS_PER_RUN=400

rm results/*
rm -r tensorboard-logs/*
rm trained-models/*

for i in $(seq 1 $NUM_RUNS)
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size 128 --regularization 0.0001 --learning-rate 0.002
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size 128 --regularization 0.001 --learning-rate 0.002
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size 128 --regularization 0.01 --learning-rate 0.002
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size 128 --regularization 0.1 --learning-rate 0.002
done
