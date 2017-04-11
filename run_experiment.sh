#!/bin/bash

while read name; do
    for iteration in {1..100}
    do
        sbatch run_trial.sh "$name" "$iteration"
    done
done <datasets.txt
