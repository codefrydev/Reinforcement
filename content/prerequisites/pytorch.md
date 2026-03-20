---
title: "PyTorch"
description: "PyTorch for RL: tensors, autograd, nn.Module, optimizers, and GPU."
date: 2026-03-10T00:00:00Z
weight: 50
draft: false
difficulty: 6
tags: ["PyTorch", "tensors", "autograd", "RL", "prerequisites"]
keywords: ["PyTorch for RL", "tensors", "autograd", "neural networks", "RL models"]
roadmap_icon: "sparkles"
roadmap_color: "rose"
roadmap_phase_label: "Phase 6 · PyTorch"
---

Used in [Preliminary: PyTorch basics](../preliminary/pytorch-basics/) and in the curriculum for DQN, policy gradients, actor-critic, PPO, and SAC. PyTorch's define-by-run style and clear autograd make it a natural fit for custom RL loss functions.

---

## Why PyTorch matters for RL

- **Tensors** — States, actions, and batches are tensors. `torch.tensor()`, `requires_grad=True`, and `.to(device)` are daily use.
- **Autograd** — Policy gradient and value losses need gradients; `backward()` and `.grad` are central.
- **nn.Module** — Q-networks, policy networks, and critics are `nn.Module` subclasses; parameters are collected for optimizers.
- **Optimizers** — `torch.optim.Adam`, `zero_grad()`, `loss.backward()`, `optimizer.step()`.
- **Device** — Move model and data to GPU with `.to(device)` for faster training.

---

## Core concepts with examples

### Tensors and gradients

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # 4.0
```

### Batches and shapes

```python
# Batch of 32 states, 4 features (e.g. CartPole)
states = torch.randn(32, 4)
# Linear layer: 4 -> 64
W = torch.randn(4, 64, requires_grad=True)
out = states @ W   # (32, 64)
```

### Simple MLP with nn.Module

```python
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

q = QNetwork()
s = torch.randn(8, 4)   # batch 8
q_vals = q(s)           # (8, 2)
```

### Training step (e.g. MSE loss)

```python
optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
targets = torch.randn(8, 2)
loss = nn.functional.mse_loss(q_vals, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Device and CPU/GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q = q.to(device)
states = states.to(device)
```

---

## Worked examples

**Example 1 — Autograd (Exercise 1).** Create \\(x = 3.0\\) with `requires_grad=True`, compute \\(y = x^3 + 2x\\), call `y.backward()`, and verify `x.grad`.

{{< collapse summary="Solution" >}}
**Step 1:** `x = torch.tensor(3.0, requires_grad=True)`. **Step 2:** `y = x**3 + 2*x` ⇒ y = 27 + 6 = 33. **Step 3:** `y.backward()`. **Step 4:** By hand, \\(dy/dx = 3x^2 + 2\\); at x=3 that is 27+2 = **29**. So `x.grad` should be `tensor(29.)`. PyTorch’s autograd applies the chain rule; we use the same mechanism for policy gradient and value loss in RL.
{{< /collapse >}}

**Example 2 — Training step.** Given a network, a batch of inputs, and targets, perform one optimizer step (zero_grad, forward, loss, backward, step).

{{< collapse summary="Solution" >}}
**Step 1:** `optimizer.zero_grad()` to clear old gradients. **Step 2:** `pred = model(batch)` then `loss = F.mse_loss(pred, targets)`. **Step 3:** `loss.backward()` to compute gradients. **Step 4:** `optimizer.step()` to update parameters. Order matters: zero_grad → forward → loss → backward → step. In RL we do this for the critic (MSE to TD target) and for the policy (gradient ascent on return).
{{< /collapse >}}

---

## Exercises

**Exercise 1.** Create a scalar tensor \\(x = 3.0\\) with `requires_grad=True`. Compute \\(y = x^3 + 2x\\) and call `y.backward()`. Verify that `x.grad` equals \\(3x^2 + 2\\) evaluated at \\(x=3\\) (i.e. 29).

**Exercise 2.** Build a 2-layer MLP: `nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))`. Forward pass a batch of 10 states of dimension 4. Print the output shape. Then compute the mean squared error between the output and a random target tensor of shape (10, 2), call `backward()` on the loss, and confirm that the first layer's weight has non-zero gradients.

**Exercise 3.** Implement a function `epsilon_greedy(q_values, epsilon)` that takes a 1D tensor `q_values` of length \\(n\\) and returns an integer action: with probability \\(\epsilon\\) sample uniformly from \\(0..n-1\\), otherwise return `argmax`. Use `torch.rand(1).item()` for the random draw and `q_values.argmax().item()` for the greedy action. No gradients needed.

**Exercise 4.** Create a policy network that outputs **logits** for 2 actions: `nn.Linear(4, 2)`. Given a state batch of shape (8, 4), compute action probabilities with `F.softmax(logits, dim=-1)` and sample 8 actions using the probabilities (e.g. `torch.multinomial(probs, 1).squeeze(-1)`). Then compute the log-probability of those actions: `F.log_softmax(logits, dim=-1)` and gather the chosen action log-probs. Return both actions and log_probs.

**Exercise 5.** Implement a training loop: (1) create the `QNetwork` above and an Adam optimizer; (2) for 100 steps, sample random states (32, 4) and random target Q-values (32, 2); (3) compute MSE loss, backward, step; (4) every 20 steps print the loss. Confirm the loss decreases.

**Exercise 6.** Create a tensor `x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)` and compute `y = x.sum()` then `y.backward()`. What is `x.grad`? **In RL:** Summing over a batch of losses is common; gradients flow back to each element.

**Exercise 7.** Build a small network that maps state dim 4 to 2 action logits. Given a batch of states (8, 4), compute log-probs with `F.log_softmax(logits, dim=-1)`. Use `torch.gather` to select the log-prob of a given action index (e.g. actions = [0, 1, 0, 1, 0, 1, 0, 1]). Return a tensor of shape (8,). **In RL:** This is the log-probability term in the policy gradient.

**Exercise 8.** (Challenge) Implement a target network: create two identical `QNetwork` instances, `q` and `q_target`. Every 10 training steps, copy `q` parameters into `q_target` with `q_target.load_state_dict(q.state_dict())`. Train `q` with MSE to `q_target` (next-state values). **In RL:** Target networks stabilize DQN.

---

## Professor's hints

- **Always call `optimizer.zero_grad()`** before `loss.backward()`; otherwise gradients accumulate across steps and your update is wrong.
- Use `loss.backward()` then `optimizer.step()` in that order. Do not call `backward()` twice on the same graph without re-running the forward pass.
- **In RL:** Policy gradient maximizes return, so you often use `loss = -log_prob * advantage` and minimize `loss`; the minus sign turns the gradient into ascent on return.
- For reproducibility, set `torch.manual_seed(42)` and `np.random.seed(42)` at the start of training.

---

## Common pitfalls

- **Forgetting `optimizer.zero_grad()`:** Gradients add by default. Without zeroing, the second step uses gradients from step 1 + step 2, which is rarely what you want.
- **Using in-place operations on tensors that require grad:** e.g. `x.add_(1)` can break the graph. Prefer `x = x + 1` or out-of-place ops when `x.requires_grad` is True.
- **Mixing CPU and GPU tensors:** Ensure model and batch are on the same device. Use `.to(device)` consistently. Calling `model(batch)` when one is on CPU and the other on GPU raises an error.
- **Taking `.item()` or indexing before backward:** If you need a Python scalar (e.g. for logging), use `.item()` on a scalar tensor only after you are done with the computation graph, or clone/detach so backward is not affected.

---

**Docs:** [pytorch.org/docs](https://pytorch.org/docs/stable/index.html). Used heavily in Volumes 3–5 (value approximation, policy gradients, PPO, SAC).
