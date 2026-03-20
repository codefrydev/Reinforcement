---
title: "PyTorch: Building Neural Networks with nn.Module"
description: "Bridge NumPy implementations to PyTorch. Build QNetwork and PolicyNetwork with nn.Module for RL."
date: 2026-03-20T00:00:00Z
weight: 12
draft: false
difficulty: 5
tags: ["PyTorch", "nn.Module", "deep learning", "QNetwork", "dl-foundations"]
keywords: ["PyTorch nn.Module", "QNetwork PyTorch", "policy network", "autograd", "optimizer.step", "RL networks"]
roadmap_icon: "globe"
roadmap_color: "amber"
roadmap_phase_label: "Chapter 12 · PyTorch"
---

**Learning objectives**
- Understand PyTorch's `nn.Module` structure and how it differs from NumPy implementations
- Build a QNetwork and PolicyNetwork using `nn.Linear` and `nn.Sequential`
- Understand the training step: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- Map the NumPy MLP you built to its PyTorch equivalent

**Concept and real-world motivation**

PyTorch provides **automatic differentiation** (autograd) — you write the forward pass, and PyTorch computes all gradients automatically via `loss.backward()`. This replaces the hand-coded backprop you implemented in NumPy. The `nn.Module` class is the building block for all networks: it tracks parameters, enables gradient flow, and handles training vs. evaluation modes.

**This page shows PyTorch syntax.** Since PyTorch doesn't run in the browser, use the linked notebook for hands-on practice.

**In RL:** All major RL frameworks use PyTorch `nn.Module` for policies and value functions. Stable-Baselines3, CleanRL, RLlib, and most research code define their networks as `nn.Module` subclasses. Understanding this pattern lets you read any modern RL codebase.

---

## From NumPy to PyTorch: side by side

**NumPy forward pass (what you've been doing):**
```python
import numpy as np

def forward(x, W1, b1, W2, b2):
    z1 = x @ W1.T + b1
    a1 = np.maximum(0, z1)   # ReLU
    z2 = a1 @ W2.T + b2
    return z2
```

**PyTorch equivalent:**
```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

The key difference: PyTorch tracks gradients automatically. No need to write backprop by hand.

---

## QNetwork for DQN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q-network: maps state → Q-values for each action."""

    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw Q-values, no activation

# Example usage:
# q_net = QNetwork(state_dim=4, n_actions=2)
# state = torch.FloatTensor([[0.1, -0.2, 0.05, 0.3]])
# q_values = q_net(state)  # shape: (1, 2)
# best_action = q_values.argmax(dim=1).item()
```

---

## PolicyNetwork with softmax output

```python
class PolicyNetwork(nn.Module):
    """Policy network: maps state → action probabilities (discrete actions)."""

    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)  # probabilities sum to 1

    def get_action(self, state):
        """Sample action from the policy distribution."""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

---

## Training step

```python
import torch.optim as optim

# Setup
model = QNetwork(state_dim=4, n_actions=2)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# One training step (from a mini-batch)
def train_step(states, actions, targets):
    states = torch.FloatTensor(states)
    targets = torch.FloatTensor(targets)
    actions = torch.LongTensor(actions)

    # 1. Zero out previous gradients
    optimizer.zero_grad()

    # 2. Forward pass
    q_values = model(states)                         # shape: (batch, n_actions)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 3. Compute loss
    loss = F.mse_loss(q_selected, targets)

    # 4. Backprop
    loss.backward()

    # 5. Update weights
    optimizer.step()

    return loss.item()
```

The three-line pattern `zero_grad → backward → step` replaces all the hand-coded gradient math.

---

## Mapping NumPy to PyTorch

| NumPy | PyTorch |
|-------|---------|
| `W1 = np.random.randn(...)` | `nn.Linear(in, out)` |
| `z = x @ W.T + b` | `self.fc(x)` |
| `np.maximum(0, z)` | `F.relu(z)` |
| Manual gradient update | `optimizer.step()` |
| Your backprop code | `loss.backward()` |
| `W -= lr * dW` | `optim.SGD(...)` or `optim.Adam(...)` |

---

**Exercise:** NumPy equivalent — implement the forward pass of a 2-layer network matching the PyTorch QNetwork above.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Match PyTorch QNetwork: state_dim=4, hidden=64, n_actions=2\nstate_dim, hidden, n_actions = 4, 64, 2\nW1 = np.random.randn(hidden, state_dim) * 0.1\nb1 = np.zeros(hidden)\nW2 = np.random.randn(hidden, hidden) * 0.1\nb2 = np.zeros(hidden)\nW3 = np.random.randn(n_actions, hidden) * 0.1\nb3 = np.zeros(n_actions)\n\ndef relu(z): return np.maximum(0, z)\n\ndef q_network_forward(state):\n    # state: shape (state_dim,)\n    h1 = relu(W1 @ state + b1)\n    h2 = relu(W2 @ h1 + b2)\n    return W3 @ h2 + b3  # raw Q-values\n\nstate = np.array([0.1, -0.2, 0.05, 0.3])\nq_values = q_network_forward(state)\nprint('Q-values:', q_values)\nprint('Best action:', np.argmax(q_values))" height="240" >}}

**Professor's hints**
- Always call `optimizer.zero_grad()` before `loss.backward()`. Forgetting this accumulates gradients across steps and produces wrong updates.
- Use `model.eval()` and `torch.no_grad()` during evaluation to disable dropout and skip gradient tracking (saves memory and compute).
- `nn.Sequential` is great for simple feed-forward networks. Use a custom `nn.Module` subclass for anything with skip connections or multiple outputs.
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`) is commonly used in RL to prevent exploding gradients.

**Common pitfalls**
- Forgetting `optimizer.zero_grad()` — gradients accumulate by default in PyTorch.
- Calling `model.train()` vs `model.eval()` at the wrong times (dropout/batchnorm behave differently in each mode).
- Using `loss.item()` to log the scalar value (not `loss` itself, which holds a computation graph).

{{< collapse summary="Worked solution — DQN-style QNetwork" >}}
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNQNetwork(nn.Module):
    """DQN-style Q-network with 3 layers."""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.conv_out = obs_dim  # replace with CNN output if using pixels
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_head = nn.Linear(512, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.q_head(x)
```
{{< /collapse >}}

**Extra practice**

1. **Notebook practice:** Complete the PyTorch exercises in the local notebook:

{{< notebook path="dl-foundations/pytorch_nn_practice.ipynb" title="PyTorch nn.Module practice (run locally)" >}}

2. **Coding:** In the notebook, implement `train_step` for a `PolicyNetwork` using cross-entropy loss. Use `torch.distributions.Categorical` to compute the log-probability of actions.

3. **Challenge:** In the notebook, implement a target network: create a second `QNetwork` with the same architecture, and periodically copy weights using `target_net.load_state_dict(online_net.state_dict())`.

4. **Variant:** Modify `QNetwork` to output both Q-values and a value estimate (for advantage computation). This is the **dueling DQN** architecture: `Q(s,a) = V(s) + A(s,a)`.

5. **Debug:** The training step below is missing `optimizer.zero_grad()`. Describe what happens to the gradients and fix it:
{{< pyrepl code="import numpy as np\n# NumPy analog: gradient accumulation bug\n# In PyTorch this would be: missing optimizer.zero_grad()\n# Let's simulate it with NumPy\nW = np.array([1.0, 0.5, -0.3])\naccumulated_grad = np.zeros(3)\nlr = 0.01\n# Three training steps WITHOUT resetting grad\nfor step in range(3):\n    x = np.random.randn(3)\n    y_true = 1.0\n    y_pred = np.dot(W, x)\n    grad = 2 * (y_pred - y_true) * x\n    accumulated_grad += grad  # BUG: should reset each step\nW -= lr * accumulated_grad\nprint('W after 3 steps (wrong, accumulated):', W)\n# Fix: reset grad = np.zeros(3) before each step" height="220" >}}

6. **Conceptual:** Why does PyTorch's `autograd` replace the need to write backprop by hand? What does the computation graph track?

7. **Recall:** Name the three steps in every PyTorch training step. What does each one do?
