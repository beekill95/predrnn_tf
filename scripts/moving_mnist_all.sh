#!/bin/bash -i

# This should be executed with slurm srun.
model=$1
use_ss=$2

# Load any modules that your program needs.
module load deeplearning

# Run moving_mnist_predrnn .py script as .ipynb notebook
# and store the result notebook.
conda activate predrnn
cat examples/moving_mnist_all.py \
    | jupytext --to ipynb \
    | papermill - "examples/moving_mnist_all_out-$SLURM_JOBID-$model-$use_ss.ipynb" \
    -p model_save_path "./saved_models/moving_mnist_all-$SLURM_JOBID-$model-$use_ss" \
    -p model_type $model \
    -p use_reversed_sampling $use_ss
