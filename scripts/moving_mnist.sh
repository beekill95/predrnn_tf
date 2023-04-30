#!/bin/bash -i

#Load any modules that your program needs
module load deeplearning

#Run your program
conda activate predrnn
papermill examples/moving_mnist_predrnn.ipynb examples/moving_mnist_predrnn_out.ipynb
