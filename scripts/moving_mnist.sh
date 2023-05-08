#!/bin/bash -i

# This should be executed with slurm srun.

# Load any modules that your program needs.
module load deeplearning

# Run moving_mnist_predrnn .py script as .ipynb notebook
# and store the result notebook.
conda activate predrnn
cat examples/moving_mnist_predrnn.py \
    | jupytext --to ipynb \
    | papermill -p model_save_path "./saved_models/moving_mnist_predrnn_$SLURM_JOBID" \
    > "examples/moving_mnist_predrnn_out_$SLURM_JOBID.ipynb"
