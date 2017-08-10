#!/bin/bash

STORE=103
EPOCHS_PER_RUN=200
BATCH_SIZE=32

rm trained-models/*
rm results/*
rm -r tensorboard-logs/*

for REGULARIZATION in 4E-3 1E-2
do
for LEARNING_RATE in 2E-4 4E-4
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --workers 4 --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
