#!/bin/bash

STORE=103
EPOCHS_PER_RUN=1000
BATCH_SIZE=256

rm trained-models/*
rm results/*
rm -r tensorboard-logs/*

for REGULARIZATION in 1E0 2E0 4E0
do
for LEARNING_RATE in 1E-3 2E-3
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --workers 6 --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
