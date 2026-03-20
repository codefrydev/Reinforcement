---
title: "Deep Learning Foundations"
description: "Neural networks, backpropagation, and training — the deep learning foundations behind DQN, policy gradients, and modern RL."
date: 2026-03-20T00:00:00Z
draft: false
difficulty: 5
tags: ["deep learning", "neural networks", "backpropagation", "dl-foundations"]
keywords: ["neural networks", "backpropagation", "deep learning", "DQN", "policy gradient network"]
roadmap_icon: "network"
roadmap_color: "purple"
roadmap_phase_label: "DL Foundations"
---

## What this section covers

Deep learning is the technology that transformed reinforcement learning from a research curiosity into a practical tool for solving hard problems. Before AlphaGo, DQN, and PPO, RL was limited to tiny, hand-crafted state spaces. Deep neural networks changed everything by serving as powerful function approximators — able to map raw pixels to values, states to action probabilities, and observations to policies.

This section builds deep learning from the ground up, starting with the biological inspiration for artificial neurons and progressing through multi-layer networks, forward propagation, loss functions, and backpropagation. Every concept is introduced with explicit connections to RL algorithms so you always know why you are learning it.

**Topics covered:**

- From biological neurons to artificial neurons: inputs, weights, bias, activation
- The perceptron: the simplest learning rule, AND gate, XOR limitations
- Activation functions: ReLU, sigmoid, tanh, softmax — when and why
- Multi-layer perceptrons: architecture, parameter counting, solving XOR
- Forward propagation: layer-by-layer computation, intermediate activations
- Loss functions: MSE for regression, cross-entropy for classification
- Backpropagation: chain rule, computing gradients, updating weights
- Gradient descent for neural networks: learning rate, momentum, Adam
- Training a neural network: mini-batches, epochs, training loop
- Regularization: dropout, weight decay, early stopping
- Convolutional neural networks: filters, pooling, feature maps
- Batch normalization and residual connections
- The complete DQN network: putting it all together

## Why deep learning matters for RL

**DQN is just Q-learning where the Q-function is a neural network.**

That single sentence captures everything. In tabular Q-learning, we store a table Q[s, a] with one entry per (state, action) pair. This works for toy problems with a handful of states. For Atari games with 210×160 pixels, the state space is astronomically large — a table is impossible. The solution: replace the table with a neural network that takes the state as input and outputs Q-values for all actions.

| DL concept | Where it reappears in RL |
|---|---|
| Artificial neuron | Building block of all value and policy networks |
| Forward propagation | Computing Q(s,a) or π(a\|s) during inference |
| Loss function (MSE) | DQN loss: \\((r + \gamma \max_{a'} Q(s', a') - Q(s,a))^2\\) |
| Loss function (cross-entropy) | Policy gradient loss |
| Backpropagation | How Q-networks and policy networks are trained |
| ReLU activations | Standard hidden-layer activation in DQN, A3C, PPO |
| Softmax | Action probability distribution in policy networks |
| Batch normalization | Stabilizing training in deep RL |
| Convolutional layers | Processing raw pixel observations in Atari DQN |
| Gradient descent / Adam | Optimizing all modern RL networks |

Policy gradient methods go further: instead of approximating a value function, they parameterize the policy itself as a neural network π(a|s; θ) and optimize the expected return directly using gradient ascent. Actor–critic methods combine both: a policy network (actor) and a value network (critic), both trained with backpropagation.

## Pedagogical approach: NumPy first

**We implement everything in NumPy first. PyTorch is introduced via linked notebooks.**

This is intentional. Implementing a neural network forward pass in NumPy — manually computing matrix multiplications, writing the ReLU function, computing the softmax — gives you a deep understanding of what the framework does for you. When you later call `torch.nn.Linear` or `loss.backward()`, you will know exactly what is happening inside.

The in-browser pyrepl exercises use NumPy exclusively because the browser environment (Pyodide) does not support PyTorch. Every concept is fully implementable in NumPy, and the implementations here are pedagogically superior to framework code for learning purposes.

The linked JupyterLite notebooks (see each page) extend the exercises and transition to PyTorch once the concepts are solid.

## Table of contents

| # | Page | Topic |
|---|---|---|
| 1 | [Biological Inspiration](biological-inspiration) | Brain neurons → artificial neurons |
| 2 | [The Perceptron](perceptron) | Perceptron learning rule, AND, XOR limits |
| 3 | [Activation Functions](activation-functions) | ReLU, sigmoid, tanh, softmax |
| 4 | [Multi-Layer Perceptrons](mlp) | Architecture, parameter counting, XOR solved |
| 5 | [Forward Propagation](forward-propagation) | Layer-by-layer computation, batch forward pass |
| 6 | [Loss Functions](loss-functions-dl) | MSE, cross-entropy, loss landscape |
| 7 | [Backpropagation](backpropagation) | Chain rule, gradients, numerical verification |
| 8 | [Gradient Descent for NNs](gradient-descent-nn) | Learning rate, momentum, Adam |
| 9 | [Training Loop](training-loop) | Mini-batches, epochs, monitoring |
| 10 | [Regularization](regularization-dl) | Dropout, weight decay, early stopping |
| 11 | [Convolutional Neural Networks](cnn-intro) | Filters, pooling, feature maps |
| 12 | [Batch Norm and Residuals](batchnorm-residual) | Normalization, skip connections |
| 13 | [The DQN Network](dqn-network) | Putting it all together for Atari |

## Quick-start guide

1. **Complete pages in order.** Each page builds on the previous one. The concepts are cumulative.
2. **Do every pyrepl exercise.** They run in your browser — no setup needed. The struggle of implementing in NumPy is where the understanding happens.
3. **Check worked solutions** only after a genuine attempt.
4. **Use the extra practice items.** Debug exercises (item 5) are especially valuable — recognizing broken code trains the same skill as writing correct code.
5. **Open the JupyterLite notebooks** for extended practice and PyTorch equivalents.

**Estimated time:** 2–4 hours per page. The full section takes approximately 30–50 hours.

## Assessment checkpoints

- **After page 3** — [Checkpoint A: Neurons and Activations](../assessment) — Can you implement a neuron and all four activations from scratch in NumPy?
- **After page 6** — [Checkpoint B: Forward Pass and Loss](../assessment) — Can you implement a full forward pass and compute MSE and cross-entropy?
- **After page 9** — [Checkpoint C: Backprop and Training](../assessment) — Can you implement backpropagation and a training loop from scratch?
- **After page 13** — [Checkpoint D: DQN Architecture](../assessment) — Can you describe the DQN network architecture and explain why each component is needed?
