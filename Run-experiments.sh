#!/bin/bash
for store in 103 109 115 151 443
do
./Cassian.py --store $store --fetch
./Cassian.py --store $store --train --epochs 40
./Cassian.py --store $store --predict
done
