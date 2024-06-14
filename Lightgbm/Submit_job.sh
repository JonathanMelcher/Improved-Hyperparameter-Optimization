#!/bin/bash

# Generate an array of starting batch values from 0 to 32
start_batches=($(seq 0 32))


# Loop through starting batch values and submit jobs
for start_batch in "${start_batches[@]}"; do
    qsub -v START_BATCH="$start_batch" LGBM_analysis.sh
done
