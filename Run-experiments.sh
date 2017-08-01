#!/bin/bash

STORE=103
NUM_RUNS=4
EPOCHS_PER_RUN=400
BATCH_SIZE=128
LEARNING_RATE=2E-4

rm trained-models/*
rm results/*

for i in $(seq 1 $NUM_RUNS)
do
for REGULARIZATION in 1E-5 2E-5 4E-5 1E-4
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
