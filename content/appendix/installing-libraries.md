---
title: "How to Install Numpy, Scipy, Matplotlib, Pandas, IPython, Theano, and TensorFlow"
description: "Install the main libraries used in the RL curriculum (and optional Theano/TensorFlow)."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["installation", "FAQ", "Python", "libraries"]
keywords: ["NumPy", "Matplotlib", "Pandas", "PyTorch", "TensorFlow", "install"]
weight: 7
roadmap_icon: "database"
roadmap_color: "indigo"
roadmap_phase_label: "Libraries"
---

**Learning objectives**

- Install NumPy, SciPy, Matplotlib, Pandas, and IPython (or Jupyter) for the curriculum.
- Optionally install Theano or TensorFlow if you follow exercises that use them; the curriculum primarily uses PyTorch for deep RL.

## Core libraries (required for early volumes)

**NumPy:** `pip install numpy`  
Used for arrays, random numbers, and numerical operations in bandits, MDPs, and tabular methods.

**Matplotlib:** `pip install matplotlib`  
Used for plotting learning curves, value functions, and heatmaps.

**Pandas:** `pip install pandas`  
Used in some exercises and the stock trading project for data handling.

**IPython / Jupyter:** `pip install ipython jupyter`  
Optional; useful for interactive experiments and notebooks.

**SciPy:** `pip install scipy`  
Optional; used in some scientific or optimization exercises.

## Deep learning and RL environments

**PyTorch:** Preferred in this curriculum for deep RL (DQN, policy gradients).  
- CPU only: `pip install torch`  
- With CUDA: see [pytorch.org](https://pytorch.org/get-started/locally/) for your OS and GPU.

**TensorFlow:** Alternative to PyTorch.  
- `pip install tensorflow` (or `tensorflow-gpu` for GPU).  
Some exercises may be written for PyTorch; the concepts transfer.

**Gym / Gymnasium:** For RL environments (CartPole, MountainCar, Blackjack).  
- `pip install gym` (legacy) or `pip install gymnasium` (maintained fork).  
Code in the curriculum may use `import gym`; Gymnasium is API-compatible for most basic usage.

## Theano

**Theano** is largely deprecated. The curriculum does not require it; we use PyTorch (or TensorFlow) for neural networks. If an older resource references Theano, you can skip it or substitute PyTorch/TensorFlow.

## Using a virtual environment

Install the above inside a [conda environment](anaconda-setup/) or `python -m venv venv` and `source venv/bin/activate` (Linux/macOS) so your system Python stays clean. Then run `pip install ...` as above.

See [Setting Up Your Environment](setting-up-environment/) and [Prerequisites](../prerequisites/) for how these libraries are used in the curriculum.
