---
title: "DL Foundations Review & Bridge to RL"
description: "Review deep learning and see why RL needs neural networks — the bridge to DQN and policy gradients."
date: 2026-03-20T00:00:00Z
weight: 100
draft: false
difficulty: 5
tags: ["deep learning", "review", "bridge to RL", "DQN", "policy gradients", "dl-foundations"]
keywords: ["DL foundations review", "bridge to RL", "DQN neural network", "policy gradient neural network", "deep RL intro"]
roadmap_icon: "brain"
roadmap_color: "indigo"
roadmap_phase_label: "Review & Bridge"
---

You have completed DL Foundations. This page reviews the key ideas and shows exactly why RL needs neural networks — bridging to DQN and policy gradients.

---

## DL Foundations Recap Quiz

Five questions to confirm your understanding. Answer before revealing each collapse.

---

**Q1.** What is the role of activation functions in neural networks?

{{< collapse summary="Answer" >}}
Activation functions introduce **non-linearity**. Without them, any number of stacked linear layers collapses to a single linear transformation \\(W'x + b'\\). Activation functions like ReLU, sigmoid, and tanh allow the network to represent complex, non-linear mappings. In practice, ReLU (`max(0,z)`) is the default for hidden layers because it doesn't cause vanishing gradients and is computationally cheap.
{{< /collapse >}}

---

**Q2.** What does backpropagation compute?

{{< collapse summary="Answer" >}}
Backpropagation computes the **gradient of the loss with respect to every weight and bias** in the network. It applies the **chain rule** from the output layer back to the first layer: \\(\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}\\). The result is a set of gradient tensors (one per parameter) used by the optimizer to update the weights.
{{< /collapse >}}

---

**Q3.** What is the difference between MSE loss and cross-entropy loss?

{{< collapse summary="Answer" >}}
- **MSE** \\(= \frac{1}{n}\sum(y_i - \hat{y}_i)^2\\) measures squared distance. Use for **regression** (predicting continuous values, like Q-values in DQN).
- **Cross-entropy** \\(= -\sum y_i \log(\hat{y}_i)\\) measures prediction quality for **probability distributions**. Use for **classification** and for policy outputs (probability over actions).

In RL: DQN's TD loss uses MSE; REINFORCE uses log-probability (related to cross-entropy) for the policy gradient.
{{< /collapse >}}

---

**Q4.** Why does adding more layers help with non-linear patterns?

{{< collapse summary="Answer" >}}
Each layer learns to combine features from the previous layer into higher-level representations. A 1-layer network can only learn simple non-linearities; a 2-layer network can approximate any continuous function (universal approximation theorem). Deeper networks learn hierarchical features: edges → textures → shapes → objects (for images); raw positions → velocity → dynamics → policy (for RL states).
{{< /collapse >}}

---

**Q5.** In 3 sentences, explain forward propagation.

{{< collapse summary="Answer" >}}
Forward propagation computes the network's output given an input. Starting from the input layer, each layer applies a linear transformation \\(z = Wx + b\\) followed by a non-linear activation \\(a = f(z)\\), and passes the result to the next layer. The final layer's output is the network's prediction — for classification, probabilities; for regression, a scalar or vector value.
{{< /collapse >}}

---

## What RL Adds to Deep Learning

| | Supervised Deep Learning | Deep RL |
|---|---|---|
| **Data source** | Fixed labeled dataset | Agent's own experience (collected during training) |
| **Labels** | Human-provided | Rewards from the environment (often sparse) |
| **Loss function** | MSE or cross-entropy | TD error (DQN), policy gradient (REINFORCE/PPO) |
| **Training stability** | Generally stable | Often unstable (correlated data, moving targets) |
| **Exploration** | Not needed | Critical — must balance exploration and exploitation |
| **Dataset size** | Fixed upfront | Grows as agent collects more experience |

**Why instability?** In supervised learning, targets are fixed. In RL, the target \\(r + \gamma \max_a Q(s', a)\\) changes as the Q-network improves. This is like chasing a moving goalposts. DQN addresses this with:
- **Target network:** A frozen copy of the Q-network used to compute targets
- **Replay buffer:** Stores past transitions and samples random mini-batches (breaks correlations)

Both are direct consequences of the instability that arises when the "dataset" and the "labels" both depend on the current network.

---

## Bridge Exercise

You know how to train a neural network on fixed data. Now imagine the data changes as you train — that is exactly what happens in RL.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Simulate a small RL training scenario\n# State: 4-dim, Action: 2 possible, Discount: 0.9\nstate_dim, n_actions = 4, 2\ngamma = 0.9\n\n# Generate 50 random (state, action, reward, next_state) transitions\nn = 50\nstates = np.random.randn(n, state_dim)\nactions = np.random.randint(0, n_actions, n)\nrewards = np.random.randn(n)  # random rewards\nnext_states = np.random.randn(n, state_dim)\n\n# Initialize Q-network: 4 -> 8 -> 2\nW1 = np.random.randn(8, state_dim) * 0.1\nb1 = np.zeros(8)\nW2 = np.random.randn(n_actions, 8) * 0.1\nb2 = np.zeros(n_actions)\n\nrelu = lambda z: np.maximum(0, z)\n\ndef q_forward(x):\n    return relu(x @ W1.T + b1) @ W2.T + b2\n\nlr = 0.01\nlosses = []\n\nfor step in range(100):\n    # Compute Q(s, a) for current network\n    q_vals = q_forward(states)  # (50, 2)\n    q_sa = q_vals[np.arange(n), actions]  # Q for chosen action\n\n    # TD targets: r + gamma * max Q(s', a') — these CHANGE as network updates!\n    with_next = q_forward(next_states)  # (50, 2)\n    td_targets = rewards + gamma * with_next.max(axis=1)\n\n    # MSE loss\n    loss = np.mean((q_sa - td_targets) ** 2)\n    losses.append(loss)\n\n    # Backprop (simplified: gradient w.r.t. output layer only)\n    delta = 2 * (q_sa - td_targets) / n\n    dq = np.zeros_like(q_vals)\n    dq[np.arange(n), actions] = delta\n    # Layer 2 gradient\n    a1 = relu(states @ W1.T + b1)\n    dW2 = dq.T @ a1; db2 = dq.sum(0)\n    W2 -= lr * dW2; b2 -= lr * db2\n\nprint(f'Initial loss: {losses[0]:.4f}')\nprint(f'Final loss:   {losses[-1]:.4f}')\nprint('Note: targets changed with network — this instability is why DQN needs a target network')" height="300" >}}

{{< collapse summary="Worked solution and key insight" >}}
The bridge exercise shows the fundamental challenge of deep RL: the targets `r + gamma * max Q(s', a')` depend on the **same network you are training**. As the network updates, the targets shift — making training unstable.

**DQN's fix:** Maintain a second "target network" with parameters \\(\theta^-\\) (updated only every \\(C\\) steps). Use it for targets: \\(y_i = r + \gamma \max_a Q(s', a; \theta^-)\\). Now targets are stable for \\(C\\) steps at a time.

```python
# Target network (NumPy version):
W1_target, b1_target = W1.copy(), b1.copy()
W2_target, b2_target = W2.copy(), b2.copy()

# Every C steps:
# W1_target, b1_target = W1.copy(), b1.copy()
# W2_target, b2_target = W2.copy(), b2.copy()
```
{{< /collapse >}}

---

## Ready for RL?

Check each box before continuing:

- [ ] I can implement forward propagation for a 2-layer network in NumPy
- [ ] I understand what backpropagation computes (gradient of loss w.r.t. all weights)
- [ ] I implemented a training loop with loss tracking (forward → loss → backprop → update)
- [ ] I understand why non-linear activations are necessary
- [ ] I know when to use MSE vs cross-entropy loss
- [ ] I understand the difference between supervised learning and deep RL (moving targets, exploration)

If all boxes are checked: **continue to RL.**

**Next steps:**

- [Prerequisites: PyTorch for RL](../../prerequisites/pytorch/) — practical PyTorch for RL implementations
- [Curriculum Volume 1: Mathematical Foundations](../../curriculum/volume-01/) — MDPs, Bellman equations, value functions

If any box is unchecked, return to the specific DL Foundations page covering that topic.
