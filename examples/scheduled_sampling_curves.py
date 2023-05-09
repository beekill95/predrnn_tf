# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     notebook_metadata_filter: title
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import matplotlib.pyplot as plt

from predrnn_tf.scheduled_sampling import (
    ExponentialScheduledSamplingLayer,
    LinearScheduledSamplingLayer,
    SigmoidScheduledSamplingLayer,
)


# %% [markdown]
# # Visualize Scheduled Sampling Curves
# ## Linear Scheduled Sampling

# %%
epsilon_s = 0.0
epsilon_e = 1.0
alpha = 1e-4
iterations = list(range(10000))

# %%
def get_epsilon_k(sampling_layer, iterations: int):
    sampling_layer.iterations = iterations
    return sampling_layer.epsilon_k


def plot_results(iterations, epsilon_k, title):
    # Plot the result.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(iterations, epsilon_k)
    ax.set_title(title)
    ax.set_ylim(0., 1.)
    fig.tight_layout()


linear_sampling = LinearScheduledSamplingLayer(
    None, epsilon_s=epsilon_s, epsilon_e=epsilon_e, alpha=alpha, reversed_iterations_start=5000)
linear_epsilon_k = [get_epsilon_k(linear_sampling, i) for i in iterations]
plot_results(iterations, linear_epsilon_k, 'Linear Scheduled Sampling')

# %% [markdown]
# ## Exponential Scheduled Sampling

# %%
alpha = 1e3
expo_sampling = ExponentialScheduledSamplingLayer(
    None, epsilon_s=epsilon_s, epsilon_e=epsilon_e, alpha=alpha)
expo_epsilon_k = [get_epsilon_k(expo_sampling, i) for i in iterations]
plot_results(iterations, expo_epsilon_k, 'Exponential Scheduled Sampling')

# %% [markdown]
# ## Sigmoid Scheduled Sampling

# %%
beta = 5000
sigmoid_sampling = SigmoidScheduledSamplingLayer(
    None, epsilon_s=epsilon_s, epsilon_e=epsilon_e, alpha=alpha, beta=beta, reversed_iterations_start=5000)
sigmoid_epsilon_k = [get_epsilon_k(sigmoid_sampling, i) for i in iterations]
plot_results(iterations, sigmoid_epsilon_k, 'Sigmoid Scheduled Sampling')
