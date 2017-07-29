#!/bin/bash

STORE=103
NUM_RUNS=10
EPOCHS_PER_RUN=200

for i in $(seq 1 $NUM_RUNS)
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN --batch_size 256
done
