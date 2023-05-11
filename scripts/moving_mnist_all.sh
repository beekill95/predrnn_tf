#!/bin/bash -i

# This should be executed with slurm srun.
model=$1
use_ss=$2

# Load any modules that your program needs.
module load deeplearning

# Run moving_mnist_predrnn .py script as .ipynb notebook
# and store the result notebook.
conda activate predrnn
cat examples/moving_mnist_predrnn_all.py \
    | jupytext --to ipynb \
    | papermill - "examples/moving_mnist_predrnn_all_out_$model_"$use_ss"_$SLURM_JOBID.ipynb" \
    -p model_save_path "./saved_models/moving_mnist_predrnn_all_$model_"$use_ss"_$SLURM_JOBID" \
    -p model_type $model \
    -p use_reversed_sampling $use_ss
