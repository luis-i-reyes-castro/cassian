#!/bin/bash

STORE=443
NUM_RUNS=10
EPOCHS_PER_RUN=10

for i in $(seq 1 $NUM_RUNS)
do
./Cassian.py --store $STORE --train --epochs $EPOCHS_PER_RUN
done
