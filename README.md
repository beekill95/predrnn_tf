# PredRNN implementation using Tensorflow.

This is an implementation of PredRNN using Tensorflow.
The implementation is based on:

* [PredRNN: recurrent neural networks for predictive learning using spatiotemporal LSTMs](https://dl.acm.org/doi/abs/10.5555/3294771.3294855)
* [PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504)
* [Official implementation in Pytorch](https://github.com/thuml/predrnn-pytorch)
* [kami93/PredRNN](https://github.com/kami93/PredRNN)

Specifically, this repo implements the second paper:
(stacked) spatial temporal cell and reversed scheduled sampling.
There is an implementation for decouple loss;
however, due to a bug with Keras's `add_loss()` function,
enabling the loss will raise error during runtime.

## Accuracy

Performance (MSE) on Moving MNIST dataset
(results produced by running `scripts/moving_mnist_all_batch.sh 5`)_

|                                        | Run #1  | Run #2  | Run #3  | Run #4  | Run #5  | Mean     | Std             |
|----------------------------------------|---------|---------|---------|---------|---------|----------|-----------------|
| ConvLSTM                               | 0.03618 | 0.00951 | 0.04391 | 0.00956 | 0.03485 | 0.026802 |   0.01613769407 |
| PredRNN without Scheduled Sampling     | 0.00705 | 0.00728 | 0.00703 | 0.00723 | 0.00733 | 0.007184 | 0.0001363084737 |
| PredRNN with Linear Scheduled Sampling | 0.00798 | 0.00741 | 0.01012 | 0.01468 | 0.00822 | 0.009682 |   0.00297355343 |

Performance (MSE) on Moving MNIST dataset with different scheduled sampling strategies.

|                                         | Run #1  | Run #2  | Run #3  | Run #4  | Run #5  | Mean     | Std             |
|-----------------------------------------|---------|---------|---------|---------|---------|----------|-----------------|
| PredRNN with Linear Scheduled Sampling  | 0.01254 | 0.01095 | 0.01231 | 0.01032 | 0.01071 | 0.011366 | 0.0009958564154 |
| PredRNN with Expo Scheduled Sampling    | 0.00816 | 0.00868 | 0.01382 | 0.01027 | 0.01094 | 0.010374 |  0.002234810954 |
| PredRNN with Sigmoid Scheduled Sampling | 0.00938 | 0.00999 | 0.00797 | 0.01007 | 0.00775 | 0.009032 |  0.001105404903 |

## Installation

The repository can be installed as module using either `pip` or `poetry`.

## Development

The repo uses `poetry` to manage dependencies.
To install all the development dependencies, use `poetry install`.

## Examples

In the folder `examples`,
there are some notebooks that create PredRNN model
and use it on Moving MNIST dataset. 

These examples can be opened as Jupyter notebooks using Jupytext.
These development dependencies will be installed along with
required dependencies with `poetry install`.

Since these examples are resource-intensive and time-consuming,
it is recommended to run these examples in GPU clusters.
In the `scripts` folder,
there are some scripts to run these in IU's Carbornate GPU clusters
(you will authorization to use these clusters, of course!).
Specifically,

* `sbatch scripts/moving_mnist.sbatch` will schedule a batch job to run
example `examples/moving_mnist_predrnn.py`.
The output notebook will be `examples/moving_mnist_predrnn_<job_id>.ipynb`
and the model will be saved as `saved_models/moving_mnist_predrnn_<job_id>`.
* Similarly for `sbatch scripts/moving_mnist_ss.sbatch`.
* `scripts/moving_mnist_all_batch.sh <nb_of_runs>` will submit a series
of jobs (3 * <nb_of_runs>) to the GPU clusters.
There will be <nb_of_runs> jobs for each combination of PredRNN/ConvLSTM
and with/without scheduled sampling.
The goal of this script is to show whether PredRNN is better than ConvLSTM
at predicting future frames for Moving MNIST dataset.
