#!/bin/bash -i

# This should be executed with slurm srun.

# Load any modules that your program needs
module load deeplearning

# Run your program
conda activate predrnn
papermill \
    examples/moving_mnist_predrnn.ipynb \
    "examples/moving_mnist_predrnn_out_$SLURM_JOBID.ipynb" \
    -p model_save_path "./saved_models/moving_mnist_predrnn_$SLURM_JOBID"
