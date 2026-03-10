---
title: "Chapter 22: Artificial Neural Networks for RL"
description: "Two-hidden-layer PyTorch network for Q-values; MSE loss."
date: 2026-03-10T00:00:00Z
weight: 22
draft: false
tags: ["neural networks", "PyTorch", "Q-values", "curriculum"]
keywords: ["neural networks for RL", "PyTorch", "Q-values", "MSE loss"]
---

**Learning objectives**

- Build a feedforward neural network that maps state to Q-values (one output per action) in PyTorch.
- Implement the forward pass and an MSE loss between predicted Q-values and targets.
- Understand how this network will be used in DQN (next chapter): TD target and gradient update.

**Concept and real-world RL**

**Neural networks** as function approximators let us represent \\(Q(s,a)\\) (or \\(Q(s)\\) with one output per action) for high-dimensional or continuous state spaces. The network takes the state (and optionally the action) as input and outputs values; we train it by minimizing TD error (e.g. MSE between predicted Q and target \\(r + \\gamma \\max_{a'} Q(s',a')\\)). This is the core of **Deep Q-Networks (DQN)** and many other deep RL algorithms. In practice, we use MLPs for low-dim state (e.g. CartPole) and CNNs for images (e.g. Atari).

**Exercise:** Build a simple neural network in PyTorch with two hidden layers (64 units each, ReLU) that takes a state vector and outputs Q-values for 2 actions. Write the forward pass and a loss function using MSE.

**Professor's hints**

- Input dim = state dim (e.g. 4 for CartPole). Output dim = number of actions (2). Architecture: `Linear(4, 64)` → ReLU → `Linear(64, 64)` → ReLU → `Linear(64, 2)`.
- Forward: `q_values = model(states)` where `states` has shape (batch, 4). Loss: `F.mse_loss(q_values, targets)` where `targets` has shape (batch, 2). Do not use `requires_grad` on targets when computing loss for DQN (targets are constants in the loss).
- Test: create a batch of random states, forward pass, check output shape (batch, 2). Create random targets, compute MSE, call `loss.backward()`, and verify that model parameters have gradients.

**Common pitfalls**

- **Output dimension:** One output per action (for discrete actions). So for 2 actions, output dim = 2. Do not output a single scalar unless you are using a different parameterization (e.g. state and action as input).
- **Target gradients:** When you compute `loss = mse_loss(q_pred, target)`, `target` should not require grad (use `.detach()` on the target when you build it from the target network in DQN).
- **Device:** Keep model and tensors on the same device (CPU or GPU). Use `model.to(device)` and `states.to(device)`.

{{< collapse summary="Worked solution (warm-up: batch shapes)" >}}
**Warm-up:** For state dim 4 and 3 actions, what are the shapes of the input tensor and the output tensor for a batch of 32? **Answer:** Input: (32, 4)—batch size 32, state dimension 4. Output: (32, 3)—batch size 32, one Q-value per action (3 actions). So the network maps each state in the batch to a 3-vector of Q(s,a) for a=0,1,2. In PyTorch we use this for DQN: `q = net(states)` gives shape (batch, n_actions).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For state dim 4 and 3 actions, what are the shapes of the input tensor and the output tensor for a batch of 32?
2. **Coding:** Build a 2-layer MLP in PyTorch that takes state (4,) and outputs Q(s,a) for 2 actions. Forward pass with batch size 16. Print output shape.
3. **Challenge:** Add a method to the network that, given a state, returns the greedy action (argmax over Q-values). Use it in a short loop to run one episode of CartPole with a random (untrained) network and report the total reward.
