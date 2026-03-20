---
title: "Volume 3 Drills — Function Approximation & DQN"
description: "15 short drill problems for Volume 3: linear FA, semi-gradient TD, DQN, replay buffer, target network, Double DQN, and dueling networks."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 7
weight: 99
tags: ["drills", "volume 3", "DQN", "function approximation", "replay buffer", "target network", "Double DQN", "dueling", "practice"]
roadmap_color: "green"
roadmap_icon: "chart"
roadmap_phase_label: "Vol 3 · Drills"
---

{{< notebook path="volume-03/vol03_drills.ipynb" title="Open drills notebook (interactive)" >}}

Short problems for Volume 3. Aim for under 5 minutes per problem. All solutions are in collapsible sections.

---

## Recall (R) — State definitions and rules

**R1.** Write the linear function approximation formula for V(s; w). What is φ(s)?

{{< collapse summary="Answer" >}}
**Linear FA:** V(s; w) = **w · φ(s)** = Σ_i w_i φ_i(s).

φ(s) is the **feature vector** (also called the feature representation or basis function evaluation) for state s. Each element φ_i(s) is a feature — e.g. position, velocity, polynomial term, tile coding activation. The weights w are learned; φ(s) is fixed (hand-designed or learned separately).

The key property: V is linear in **w** (not necessarily in s).
{{< /collapse >}}

---

**R2.** What is the semi-gradient TD(0) update for linear FA? How does it differ from true gradient descent?

{{< collapse summary="Answer" >}}
**Semi-gradient TD(0) update:**

w ← w + α [R_{t+1} + γ V(S_{t+1}; w) − V(S_t; w)] · ∇_w V(S_t; w)

For linear FA: ∇_w V(S_t; w) = φ(S_t), so: w ← w + α δ_t φ(S_t).

**Why semi-gradient?** The TD target R_{t+1} + γ V(S_{t+1}; w) itself depends on w. True gradient descent would differentiate through the target too, giving an extra term. Semi-gradient **stops the gradient through the target** (treats it as a fixed label). This is biased but converges reliably for linear FA; true gradient TD exists but is more complex.
{{< /collapse >}}

---

**R3.** What are the two key components that make DQN stable? What problem does each solve?

{{< collapse summary="Answer" >}}
1. **Replay buffer (experience replay):** Stores (s, a, r, s', done) tuples and samples random mini-batches for training. Solves: (1) **correlated updates** — consecutive transitions are highly correlated; random sampling breaks this, making updates more like i.i.d. supervised learning. (2) **data efficiency** — each experience can be reused many times.

2. **Target network:** A separate copy of the Q-network with weights θ⁻ that are updated less frequently (e.g. every C steps). The TD target uses θ⁻ instead of the current θ. Solves: **moving target problem** — without a frozen target, updating Q toward a target that also changes rapidly causes instability (the "chasing your own tail" problem).
{{< /collapse >}}

---

**R4.** Write the DQN TD target y for a non-terminal transition (s, a, r, s').

{{< collapse summary="Answer" >}}
**DQN TD target:**

y = r + γ · max_{a'} Q(s', a'; θ⁻)

where θ⁻ are the **target network** weights (frozen for C steps). For a terminal transition: y = r.

The network is trained to minimize (y − Q(s, a; θ))², using gradient descent only through Q(s,a;θ) — not through y (semi-gradient / stop-gradient on target).
{{< /collapse >}}

---

**R5.** What is the Double DQN improvement? What bias does vanilla DQN have that Double DQN corrects?

{{< collapse summary="Answer" >}}
**Vanilla DQN bias:** The target max_{a'} Q(s', a'; θ⁻) uses the **same network** to both *select* the best action and *evaluate* its value. Because Q-values have estimation noise, taking the max over noisy estimates is a biased estimator — it **overestimates** Q-values (maximization bias).

**Double DQN fix:** Decouple selection and evaluation:

y = r + γ · Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)

- **Select** the best action using the **online network** θ (current weights).
- **Evaluate** that action using the **target network** θ⁻.

This reduces overestimation bias, leading to more accurate value estimates and better policies.
{{< /collapse >}}

---

## Compute (C) — Numerical exercises

**C1.** Compute V(s; w) for a state with feature vector φ(s) = [1, 0.5, -0.3] and weights w = [2.0, 1.0, -1.0].

{{< pyrepl code="phi = [1.0, 0.5, -0.3]\nw = [2.0, 1.0, -1.0]\n\nV = sum(wi * phi_i for wi, phi_i in zip(w, phi))\nprint(f'V(s; w) = {V:.4f}')   # 2*1 + 1*0.5 + (-1)*(-0.3) = 2.8" height="160" >}}

{{< collapse summary="Answer" >}}
V(s; w) = 2.0×1 + 1.0×0.5 + (−1.0)×(−0.3) = 2.0 + 0.5 + 0.3 = **2.8**.
{{< /collapse >}}

---

**C2.** Apply one semi-gradient TD(0) update. Current w = [2.0, 1.0, -1.0], φ(s) = [1, 0.5, -0.3], r = 0, φ(s') = [0, 1, 0], γ = 0.9, α = 0.01.

{{< pyrepl code="import math\n\nw = [2.0, 1.0, -1.0]\nphi_s  = [1.0, 0.5, -0.3]\nphi_s_next = [0.0, 1.0, 0.0]\nr = 0\nalpha = 0.01\ngamma = 0.9\n\nV_s      = sum(wi*pi for wi,pi in zip(w, phi_s))\nV_s_next = sum(wi*pi for wi,pi in zip(w, phi_s_next))\ntd_error = r + gamma * V_s_next - V_s\n\nw_new = [wi + alpha * td_error * phi_i\n         for wi, phi_i in zip(w, phi_s)]\n\nprint(f'V(s)      = {V_s:.4f}')        # 2.8\nprint(f'V(s_next) = {V_s_next:.4f}')   # 1.0\nprint(f'TD error  = {td_error:.4f}')   # 0.9 - 2.8 = -1.9\nprint(f'w_new     = {[round(v,4) for v in w_new]}')" height="260" >}}

{{< collapse summary="Answer" >}}
V(s) = 2.8, V(s') = 1.0. TD error δ = 0 + 0.9×1.0 − 2.8 = **−1.9**.

Δw = α × δ × φ(s) = 0.01 × (−1.9) × [1, 0.5, −0.3] = [−0.019, −0.0095, 0.0057].

w_new = [2.0−0.019, 1.0−0.0095, −1.0+0.0057] = **[1.981, 0.9905, −0.9943]**.
{{< /collapse >}}

---

**C3.** Compute the DQN target y for a non-terminal transition: r = 1, γ = 0.99. Target network Q-values for s': [0.3, 0.7, 0.2, 0.5].

{{< pyrepl code="r = 1\ngamma = 0.99\nQ_target_s_next = [0.3, 0.7, 0.2, 0.5]\n\ny = r + gamma * max(Q_target_s_next)\nprint(f'Best Q(s_next) = {max(Q_target_s_next)}')\nprint(f'DQN target y   = {y:.4f}')   # 1 + 0.99*0.7 = 1.693" height="160" >}}

{{< collapse summary="Answer" >}}
y = 1 + 0.99 × max([0.3, 0.7, 0.2, 0.5]) = 1 + 0.99 × 0.7 = **1.693**.
{{< /collapse >}}

---

**C4.** Compute the Double DQN target for the same transition. Online network Q-values for s': [0.4, 0.6, 0.8, 0.1]. Target network Q-values for s': [0.3, 0.7, 0.2, 0.5].

{{< pyrepl code="r = 1\ngamma = 0.99\nQ_online_s_next  = [0.4, 0.6, 0.8, 0.1]   # used to SELECT action\nQ_target_s_next  = [0.3, 0.7, 0.2, 0.5]   # used to EVALUATE action\n\n# Step 1: select best action using online network\nbest_action = Q_online_s_next.index(max(Q_online_s_next))\n# Step 2: evaluate that action using target network\ny_double = r + gamma * Q_target_s_next[best_action]\n\nprint(f'Best action (online): {best_action}')   # action 2\nprint(f'Q_target[{best_action}]  = {Q_target_s_next[best_action]}')\nprint(f'Double DQN y = {y_double:.4f}')   # 1 + 0.99*0.2 = 1.198\nprint(f'Vanilla DQN  y = {r + gamma * max(Q_target_s_next):.4f}')  # 1.693" height="240" >}}

{{< collapse summary="Answer" >}}
Online network selects action 2 (Q=0.8). Target network evaluates it: Q_target[2] = 0.2.

Double DQN y = 1 + 0.99 × 0.2 = **1.198** vs. vanilla DQN y = **1.693**.

The vanilla DQN overestimates: it picked the action with the highest *target* Q-value, which is inflated by noise. Double DQN is less optimistic.
{{< /collapse >}}

---

**C5.** Huber loss vs MSE for a TD error δ = 3.0. Compute both (use δ_clip = 1.0 for Huber).

{{< pyrepl code="import math\n\ndelta = 3.0\ndelta_clip = 1.0\n\nmse_loss = 0.5 * delta**2\n\n# Huber loss: 0.5*delta^2 if |delta| <= delta_clip, else delta_clip*(|delta| - 0.5*delta_clip)\nif abs(delta) <= delta_clip:\n    huber_loss = 0.5 * delta**2\nelse:\n    huber_loss = delta_clip * (abs(delta) - 0.5 * delta_clip)\n\nprint(f'MSE loss   = {mse_loss:.2f}')    # 4.5\nprint(f'Huber loss = {huber_loss:.2f}')  # 1*(3 - 0.5) = 2.5\nprint(f'Ratio (MSE/Huber) = {mse_loss/huber_loss:.2f}')  # 1.8x larger" height="220" >}}

{{< collapse summary="Answer" >}}
MSE = 0.5 × 3² = **4.5**. Huber (δ_clip=1) = 1 × (3 − 0.5) = **2.5**.

Huber is less sensitive to large TD errors. For δ=3, MSE gives a gradient of 3 while Huber gives a gradient of 1 (clipped). This makes Huber more robust to outlier transitions in the replay buffer.
{{< /collapse >}}

---

## Code (K) — Implementation

**K1.** Implement a replay buffer as a fixed-size deque that supports `push` and `sample`.

{{< pyrepl code="import random\nfrom collections import deque\n\nclass ReplayBuffer:\n    def __init__(self, capacity):\n        # TODO\n        pass\n\n    def push(self, transition):\n        \"\"\"transition = (s, a, r, s_next, done)\"\"\"\n        # TODO\n        pass\n\n    def sample(self, batch_size):\n        \"\"\"Return a random batch of transitions.\"\"\"\n        # TODO\n        pass\n\n    def __len__(self):\n        # TODO\n        pass\n\nbuf = ReplayBuffer(capacity=5)\nfor i in range(7):\n    buf.push((i, 0, float(i), i+1, False))\nprint(f'Length (max 5): {len(buf)}')   # 5\nbatch = buf.sample(3)\nprint(f'Sample size: {len(batch)}')    # 3\nprint('First transition in sample:', batch[0])" height="280" >}}

{{< collapse summary="Solution" >}}
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```
`deque(maxlen=capacity)` automatically discards the oldest entry when the buffer is full — no manual eviction needed.
{{< /collapse >}}

---

**K2.** Implement an epsilon decay schedule: start at ε_start=1.0, end at ε_end=0.05, decay over `decay_steps` steps using exponential decay.

{{< pyrepl code="import math\n\ndef get_epsilon(step, eps_start=1.0, eps_end=0.05, decay_steps=10000):\n    \"\"\"Exponential epsilon decay.\"\"\"\n    # TODO\n    pass\n\n# Test: print epsilon at steps 0, 1000, 5000, 10000\nfor step in [0, 1000, 5000, 10000, 20000]:\n    print(f'step={step:6d}: eps={get_epsilon(step):.4f}')" height="200" >}}

{{< collapse summary="Solution" >}}
```python
def get_epsilon(step, eps_start=1.0, eps_end=0.05, decay_steps=10000):
    decay = math.exp(math.log(eps_end / eps_start) / decay_steps)
    return max(eps_end, eps_start * decay**step)
```
At step 0: ε=1.0. At step 10000: ε=0.05. After that: clipped at ε_end.
{{< /collapse >}}

---

## Debug (D) — Find and fix the bug

**D1.** This DQN training loop has a critical bug. Find and fix it.

```python
def train_step(online_net, target_net, batch, gamma=0.99, optimizer=None):
    states, actions, rewards, next_states, dones = zip(*batch)

    # Compute target
    next_q = target_net(next_states)
    targets = [r + gamma * max(nq) if not d else r
               for r, nq, d in zip(rewards, next_q, dones)]

    # Compute prediction
    pred_q = online_net(states)
    pred = [pred_q[i][a] for i, a in enumerate(actions)]

    loss = mse_loss(pred, targets)
    optimizer.zero_grad()
    loss.backward()   # Bug: gradients flow through targets too!
    optimizer.step()
```

{{< collapse summary="Answer" >}}
The bug: `target_net(next_states)` computes `next_q` but the target values (`targets`) are still part of the computational graph — **gradients flow back through the target network** during `loss.backward()`. This is wrong for two reasons: (1) it updates the target network (defeating its purpose); (2) it makes training unstable.

**Fix:** Use `torch.no_grad()` when computing the target:
```python
with torch.no_grad():
    next_q = target_net(next_states)
    targets = [r + gamma * nq.max().item() if not d else r
               for r, nq, d in zip(rewards, next_q, dones)]
```
The `torch.no_grad()` context stops gradient computation entirely for the target. Alternatively use `.detach()` on `next_q`.
{{< /collapse >}}

---

**D2.** Find the bug in this target network update code:

```python
class DQNAgent:
    def __init__(self):
        self.online_net = QNetwork()
        self.target_net = QNetwork()
        self.step_count = 0
        self.update_every = 1000  # update target every 1000 steps

    def learn(self, batch):
        # ... compute loss, do gradient step ...
        self.step_count += 1
        # Update target network
        self.target_net.load_state_dict(self.online_net.state_dict())  # Bug!
```

{{< collapse summary="Answer" >}}
The bug: the target network is updated **every single step** (no `if` guard). The whole point of the target network is to be a **slowly-changing** copy of the online network. Updating it every step makes it identical to the online network — no stability benefit.

**Fix:**
```python
if self.step_count % self.update_every == 0:
    self.target_net.load_state_dict(self.online_net.state_dict())
```
Now the target network is a frozen snapshot of the online network, updated every `update_every` steps (e.g. every 1000 steps), providing a stable regression target.
{{< /collapse >}}

---

## Challenge (X)

**X1.** Implement a minimal DQN training loop (without PyTorch — using pure Python linear networks) on the 3×3 gridworld. Use: replay buffer (capacity 1000), target network (update every 50 steps), ε-greedy (exponential decay), mini-batch size 32. Train for 2000 steps and report final policy.

{{< pyrepl code="import random, math\nfrom collections import deque\nrandom.seed(42)\n\n# 3x3 gridworld\nACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]\n\ndef env_step(s, a):\n    r_new = s[0] + ACTIONS[a][0]\n    c_new = s[1] + ACTIONS[a][1]\n    s_next = (max(0,min(2,r_new)), max(0,min(2,c_new)))\n    if s_next == (2,2):\n        return s_next, 1.0, True\n    return s_next, -0.01, False\n\ndef state_features(s):\n    \"\"\"Simple one-hot encoding: 9 features\"\"\"\n    idx = s[0]*3 + s[1]\n    return [1.0 if i == idx else 0.0 for i in range(9)]\n\n# TODO: implement linear Q-network, replay buffer, DQN loop\n# Hint: Q(s,a) = w[a] . phi(s), gradient: dL/dw[a] = 2*(Q(s,a)-y)*phi(s)\nprint('Implement minimal DQN...')" height="260" >}}

{{< collapse summary="Hint" >}}
1. **Network:** weights = [[0.0]*9 for _ in range(4)] (one weight vector per action).
2. **Forward:** Q(s,a) = dot(weights[a], state_features(s)).
3. **Update:** delta = Q(s,a) - y; weights[a] = [w - lr*delta*phi for w, phi in zip(weights[a], phi_s)].
4. **ReplayBuffer:** deque(maxlen=1000), push (s,a,r,s_next,done), sample random.
5. **Target network:** separate copy of weights, updated every 50 steps by copying.
6. **Training loop:** reset env, ε-greedy action, step, push to buffer, sample batch, compute targets, update weights, decay ε.

Expected: policy converges to arrows toward (2,2) within ~1000 steps.
{{< /collapse >}}
