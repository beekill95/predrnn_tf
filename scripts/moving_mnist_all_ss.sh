#!/bin/bash -i

# This should be executed with slurm srun.
ss_type=$1

# Load any modules that your program needs.
module load deeplearning

# Run moving_mnist_predrnn .py script as .ipynb notebook
# and store the result notebook.
conda activate predrnn
cat examples/moving_mnist_all_ss.py \
    | jupytext --to ipynb \
    | papermill - "examples/moving_mnist_all_ss_out-$SLURM_JOBID-$ss_type.ipynb" \
    -p model_save_path "./saved_models/moving_mnist_all_ss-$SLURM_JOBID-$ss_type" \
    -p ss_type "$ss_type"
