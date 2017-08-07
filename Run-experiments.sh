#!/bin/bash

STORE=103
EPOCHS_PER_RUN=200
BATCH_SIZE=64

rm trained-models/*
rm results/*
rm -r tensorboard-logs/*

for LEARNING_RATE in 1E-3 2E-3 4E-3
do
for REGULARIZATION in 1E-6 1E-5 1E-4
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --workers 6 --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
