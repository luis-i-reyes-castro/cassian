#!/bin/bash

STORE=103
EPOCHS_PER_RUN=800
BATCH_SIZE=256

rm trained-models/*
rm results/*

for REGULARIZATION in 4E-6 1E-5 2E-5 4E-5 1E-4 2E-4 4E-4 1E-3
do
for LEARNING_RATE in 4E-3 2E-3 8E-3
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch-size $BATCH_SIZE --regularization $REGULARIZATION --learning-rate $LEARNING_RATE
done
done
