#!/bin/bash

STORE=103
NUM_RUNS=4
EPOCHS_PER_RUN=400
BATCH_SIZE=128
LEARNING_RATE=1E-3

rm trained-models/*
rm results/*

for i in $(seq 1 $NUM_RUNS)
do
for REGULARIZATION in 4E-6 1E-5 2E-5 4E-5 1E-4
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
