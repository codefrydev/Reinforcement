---
title: "Bridge Exercises: Python + Math + RL"
description: "15 problems combining Python, probability, and toy RL. Complete before starting Volume 1."
date: 2026-03-19T00:00:00Z
draft: false
weight: 5
tags: ["bridge exercises", "phase 2.5", "Python", "math", "RL", "practice"]
keywords: ["bridge exercises", "Python RL practice", "pre-curriculum exercises", "tabular RL beginner", "discounted return exercises"]
---

These 15 exercises sit at **Level 2.5** — after prerequisites (Python, NumPy, math) but before Volume 1. They combine the skills you have learned so far in a context that looks like real RL code. If you can complete all 15, you are ready for the curriculum.

Each exercise has an interactive REPL so you can run code in your browser.

---

## B1 — Sample from a slot machine

Simulate pulling a single bandit arm 1000 times. The arm gives reward `Normal(true_mean=0.5, std=1)`. Compute the sample mean. It should be close to 0.5.

{{< pyrepl code="import random\nrandom.seed(42)\n\ndef gauss_arm(true_mean, std=1.0):\n    # Box-Muller (no NumPy needed)\n    import math\n    u1, u2 = random.random(), random.random()\n    z = math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)\n    return true_mean + std * z\n\n# TODO: pull 1000 times, compute sample mean\npulls = []\n# ...\nprint(f'Sample mean: {sum(pulls)/len(pulls):.3f}  (true=0.5)')" height="280" >}}

{{< collapse summary="Solution" >}}
```python
import random, math
random.seed(42)

def gauss_arm(true_mean, std=1.0):
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2*math.log(u1+1e-10)) * math.cos(2*math.pi*u2)
    return true_mean + std * z

pulls = [gauss_arm(0.5) for _ in range(1000)]
print(f"Sample mean: {sum(pulls)/len(pulls):.3f}  (true=0.5)")
```
{{< /collapse >}}

**Why this matters:** Bandit algorithms estimate each arm's true mean from samples — exactly this calculation, repeated per arm.

---

## B2 — Valid moves in a grid

Write `valid_moves(state, n=4)` that returns a list of valid action indices (0=up, 1=down, 2=left, 3=right) for an n×n grid. An action is valid if it stays in bounds.

{{< pyrepl code="def valid_moves(state, n=4):\n    row, col = state\n    moves = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}\n    # TODO: return list of valid action keys\n    pass\n\nprint(valid_moves((0,0), 4))   # [1, 3]\nprint(valid_moves((2,2), 4))   # [0, 1, 2, 3]\nprint(valid_moves((3,3), 4))   # [0, 2]" height="240" >}}

{{< collapse summary="Solution" >}}
```python
def valid_moves(state, n=4):
    row, col = state
    moves = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
    return [a for a,(dr,dc) in moves.items()
            if 0 <= row+dr < n and 0 <= col+dc < n]

print(valid_moves((0,0), 4))   # [1, 3]
print(valid_moves((2,2), 4))   # [0, 1, 2, 3]
print(valid_moves((3,3), 4))   # [0, 2]
```
{{< /collapse >}}

---

## B3 — Discounted return

Implement `discounted_return(rewards, gamma)`. Verify:
- `[0,0,0,1]`, γ=0.9 → `0.9³ = 0.729`
- `[1,0,0,0]`, γ=0.9 → `1.0`
- `[1,1,1,1]`, γ=0.9 → `1 + 0.9 + 0.81 + 0.729 = 3.439`

{{< pyrepl code="def discounted_return(rewards, gamma=0.9):\n    pass\n\nprint(f'{discounted_return([0,0,0,1]):.4f}')   # 0.7290\nprint(f'{discounted_return([1,0,0,0]):.4f}')   # 1.0000\nprint(f'{discounted_return([1,1,1,1]):.4f}')   # 3.4390" height="220" >}}

{{< collapse summary="Solution" >}}
```python
def discounted_return(rewards, gamma=0.9):
    return sum(gamma**t * r for t, r in enumerate(rewards))

print(f"{discounted_return([0,0,0,1]):.4f}")   # 0.7290
print(f"{discounted_return([1,0,0,0]):.4f}")   # 1.0000
print(f"{discounted_return([1,1,1,1]):.4f}")   # 3.4390
```
{{< /collapse >}}

---

## B4 — Greedy action from a Q-table

Write `greedy_action(Q, state, n_actions=4)` that returns the best action for a state from a dict Q where keys are `(state, action)` tuples.

{{< pyrepl code="def greedy_action(Q, state, n_actions=4):\n    # Q.get((state, a), 0.0) for each a in range(n_actions)\n    pass\n\nQ = {((0,0),0): -0.2, ((0,0),1): 0.5, ((0,0),2): -0.1, ((0,0),3): 0.3}\nprint(greedy_action(Q, (0,0)))   # 1 (highest Q=0.5)" height="220" >}}

{{< collapse summary="Solution" >}}
```python
def greedy_action(Q, state, n_actions=4):
    return max(range(n_actions), key=lambda a: Q.get((state, a), 0.0))

Q = {((0,0),0):-0.2, ((0,0),1):0.5, ((0,0),2):-0.1, ((0,0),3):0.3}
print(greedy_action(Q, (0,0)))   # 1
```
{{< /collapse >}}

---

## B5 — Random walk

Simulate a 1D random walk: 5 positions (0–4). Start at position 2. Each step, go left or right with equal probability. Stop at 0 or 4. Run 2000 episodes. Print what fraction of episodes end at position 4.

{{< pyrepl code="import random\nrandom.seed(42)\n\ndef random_walk_episode(start=2):\n    pos = start\n    while pos not in (0, 4):\n        pos += 1 if random.random() < 0.5 else -1\n    return pos\n\n# TODO: run 2000 episodes, count fraction ending at 4\nresults = []\nprint(f'Fraction ending at 4: {sum(results)/len(results):.3f}')   # ~0.5" height="260" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)

def random_walk_episode(start=2):
    pos = start
    while pos not in (0, 4):
        pos += 1 if random.random() < 0.5 else -1
    return pos

results = [random_walk_episode() for _ in range(2000)]
print(f"Fraction ending at 4: {sum(r==4 for r in results)/len(results):.3f}")
```
{{< /collapse >}}

**Why this matters:** Value function estimation uses many such "episodes" to estimate how often good outcomes occur from each state.

---

## B6 — Incremental mean

Implement `incremental_mean(current_mean, n, new_value)` = `current_mean + (new_value - current_mean) / n`.

Run: pull arm 5 times with rewards `[1.2, 0.8, 1.0, 1.4, 0.6]`. Update mean incrementally. Final mean should be 1.0.

{{< pyrepl code="def incremental_mean(current_mean, n, new_value):\n    pass\n\nQ = 0.0\nfor n, r in enumerate([1.2, 0.8, 1.0, 1.4, 0.6], 1):\n    Q = incremental_mean(Q, n, r)\n    print(f'After {n} pulls: Q={Q:.3f}')\nprint(f'Final mean: {Q:.3f}')   # 1.000" height="280" >}}

{{< collapse summary="Solution" >}}
```python
def incremental_mean(current_mean, n, new_value):
    return current_mean + (new_value - current_mean) / n

Q = 0.0
for n, r in enumerate([1.2, 0.8, 1.0, 1.4, 0.6], 1):
    Q = incremental_mean(Q, n, r)
print(f"Final mean: {Q:.3f}")   # 1.000
```
{{< /collapse >}}

---

## B7 — Sample mean and variance

Given 5 reward samples `[2.1, 1.8, 2.3, 1.9, 2.4]`, compute:
- Sample mean (should be 2.1)
- Unbiased sample variance (use n-1 denominator)

{{< pyrepl code="data = [2.1, 1.8, 2.3, 1.9, 2.4]\n# TODO: mean = sum/len\n# TODO: var = sum((x-mean)**2 for x in data) / (len(data)-1)\nmean = None\nvar = None\nprint(f'Mean: {mean:.2f}, Var: {var:.4f}')" height="220" >}}

{{< collapse summary="Solution" >}}
```python
data = [2.1, 1.8, 2.3, 1.9, 2.4]
mean = sum(data) / len(data)
var = sum((x - mean)**2 for x in data) / (len(data) - 1)
print(f"Mean: {mean:.2f}, Var: {var:.4f}")   # 2.10, 0.0550
```
{{< /collapse >}}

---

## B8 — Dot product (linear value function)

Feature vector for state (2, 3) in a 5×5 grid: `[1, 2/4, 3/4]` (bias, normalized row, normalized col). Weights: `[0.1, 0.5, 0.3]`. Compute `V(s) = w · phi(s)`.

{{< pyrepl code="w = [0.1, 0.5, 0.3]\nphi = [1, 2/4, 3/4]   # features for state (2,3)\n# TODO: V_s = sum(w_i * phi_i for i in ...)\nV_s = None\nprint(f'V(s) = {V_s:.4f}')" height="200" >}}

{{< collapse summary="Solution" >}}
```python
w = [0.1, 0.5, 0.3]
phi = [1, 2/4, 3/4]
V_s = sum(wi * pi for wi, pi in zip(w, phi))
print(f"V(s) = {V_s:.4f}")   # 0.1 + 0.25 + 0.225 = 0.575
```
{{< /collapse >}}

---

## B9 — Policy as a function

An epsilon-greedy policy selects: random action with probability ε, best action (argmax Q) otherwise. Implement `eps_greedy_policy(Q_s, epsilon, n_actions)` where `Q_s` is a list indexed by action.

{{< pyrepl code="import random\nrandom.seed(0)\n\ndef eps_greedy_policy(Q_s, epsilon, n_actions):\n    pass\n\nQ_s = [0.1, 0.7, 0.3, 0.4]\n# With epsilon=0, should always return 1\nprint([eps_greedy_policy(Q_s, 0.0, 4) for _ in range(5)])   # [1,1,1,1,1]" height="240" >}}

{{< collapse summary="Solution" >}}
```python
import random

def eps_greedy_policy(Q_s, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return Q_s.index(max(Q_s))

random.seed(0)
Q_s = [0.1, 0.7, 0.3, 0.4]
print([eps_greedy_policy(Q_s, 0.0, 4) for _ in range(5)])
```
{{< /collapse >}}

---

## B10 — Bellman backup (one step)

For a 2-state MDP:
- States: A, B (both non-terminal)
- From A: go to B with prob 1, reward = 0
- From B: go to A with prob 1, reward = 1
- γ = 0.9, current V = {"A": 0.5, "B": 0.4}

Compute one Bellman backup for V(A) and V(B):
- `V_new(A) = 0 + 0.9 * V(B)` = ?
- `V_new(B) = 1 + 0.9 * V(A)` = ?

{{< pyrepl code="V = {'A': 0.5, 'B': 0.4}\ngamma = 0.9\n# TODO:\n# V_new_A = reward(A->B) + gamma * V['B']\n# V_new_B = reward(B->A) + gamma * V['A']\nV_new_A = None\nV_new_B = None\nprint(f'V_new(A) = {V_new_A:.4f}')   # 0.36\nprint(f'V_new(B) = {V_new_B:.4f}')   # 1.45" height="240" >}}

{{< collapse summary="Solution" >}}
```python
V = {"A": 0.5, "B": 0.4}
gamma = 0.9
V_new_A = 0 + gamma * V["B"]    # 0.36
V_new_B = 1 + gamma * V["A"]    # 1.45
print(f"V_new(A) = {V_new_A:.4f}")
print(f"V_new(B) = {V_new_B:.4f}")
```

**The true values:** Solving V(A) = 0.9*V(B) and V(B) = 1 + 0.9*V(A) simultaneously gives V(A) ≈ 4.74, V(B) ≈ 5.26 (the sum 0+1=1 reward is collected in every 2-step cycle, heavily discounted over infinite horizon).
{{< /collapse >}}

---

## B11 — TD(0) update

TD(0): `V(s) ← V(s) + α * [r + γ*V(s') - V(s)]`

Given: `V(A)=0.3`, transition A→B with `r=0`, `V(B)=0.5`, `α=0.1`, `γ=0.9`.

Compute `δ` (TD error) and new `V(A)`.

{{< pyrepl code="V_A = 0.3\nV_B = 0.5\nr = 0\nalpha, gamma = 0.1, 0.9\n\n# TODO: delta = r + gamma * V_B - V_A\n# V_A_new = V_A + alpha * delta\ndelta = None\nV_A_new = None\nprint(f'TD error δ = {delta:.4f}')     # 0.15\nprint(f'V(A) new   = {V_A_new:.4f}')   # 0.315" height="240" >}}

{{< collapse summary="Solution" >}}
```python
V_A = 0.3; V_B = 0.5; r = 0; alpha = 0.1; gamma = 0.9
delta = r + gamma * V_B - V_A      # 0 + 0.45 - 0.3 = 0.15
V_A_new = V_A + alpha * delta      # 0.3 + 0.015 = 0.315
print(f"TD error δ = {delta:.4f}")
print(f"V(A) new   = {V_A_new:.4f}")
```
{{< /collapse >}}

---

## B12 — Multi-armed bandit: 2000 steps

Implement a full 2000-step bandit agent: 5 arms with random true means (seed=42). Use epsilon-greedy (ε=0.1) and incremental updates. Report total reward, best arm found, and whether it matches the true best arm.

{{< pyrepl code="import random, math\nrandom.seed(42)\n\nk = 5\ntrue_means = [random.gauss(0, 1) for _ in range(k)]\nQ = [0.0] * k\nN = [0] * k\ntotal_reward = 0\nepsilon = 0.1\n\n# TODO: 2000 steps of epsilon-greedy with incremental update\n\nprint('True means:   ', [round(m,3) for m in true_means])\nprint('Q estimates:  ', [round(q,3) for q in Q])\nprint('Pull counts:  ', N)\nprint('True best arm:', true_means.index(max(true_means)))\nprint('Est best arm: ', Q.index(max(Q)))\nprint(f'Total reward: {total_reward:.1f}')" height="320" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)

k = 5
true_means = [random.gauss(0, 1) for _ in range(k)]
Q = [0.0] * k
N = [0] * k
total_reward = 0
epsilon = 0.1

for _ in range(2000):
    a = random.randrange(k) if random.random() < epsilon else Q.index(max(Q))
    r = random.gauss(true_means[a], 1)
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]
    total_reward += r

print("True best:", true_means.index(max(true_means)))
print("Est best: ", Q.index(max(Q)))
print(f"Total reward: {total_reward:.1f}")
```
{{< /collapse >}}

---

## B13 — Find the bug: off-by-one in episode

This function is supposed to compute the first-visit MC return for a list of `(state, reward)` pairs, but there is a bug. Find and fix it.

```python
def first_visit_mc(episode, target_state):
    """Return the return G from first visit to target_state."""
    states = [s for s, r in episode]
    rewards = [r for s, r in episode]
    if target_state not in states:
        return None
    t = states.index(target_state)
    G = sum(rewards[t:])   # bug: should this be rewards[t:] or rewards[t+1:]?
    return G
```

{{< pyrepl code="def first_visit_mc(episode, target_state):\n    states = [s for s, r in episode]\n    rewards = [r for s, r in episode]\n    if target_state not in states:\n        return None\n    t = states.index(target_state)\n    G = sum(rewards[t:])   # Is this correct?\n    return G\n\n# Episode: at step 0 state=A gets r=0; step 1 state=B gets r=0; step 2 state=C gets r=1\nepisode = [('A', 0), ('B', 0), ('C', 1)]\nprint(first_visit_mc(episode, 'A'))   # should be 1 (all future rewards from A)\nprint(first_visit_mc(episode, 'B'))   # should be 1 (rewards from B onwards)\nprint(first_visit_mc(episode, 'C'))   # should be 1 (reward at C)" height="300" >}}

{{< collapse summary="Answer and explanation" >}}
The code is actually **correct** for this MC formulation. `G_t = r_t + r_{t+1} + ...` includes the reward received **at** step t. `rewards[t:]` starts from the reward received when visiting `target_state`, which is the standard first-visit MC definition.

The common bug students introduce is using `rewards[t+1:]` (skipping the immediate reward at the visited state). Whether `r_t` should be included depends on the convention: some formulations use `G_t = r_{t+1} + γr_{t+2} + ...` (reward after the state). The key is consistency.

**Lesson:** Always check whether your return formula includes the reward received at the current step or only future rewards. Sutton & Barto use `G_t = R_{t+1} + γR_{t+2} + ...` (reward *after* action), so the reward at step t is `rewards[t]` in 0-indexed code.
{{< /collapse >}}

---

## B14 — Gradient step

Implement one gradient descent step for a linear value function:
- `w ← w - α * δ * φ(s)` (semi-gradient TD)
- `w = [0.5, -0.3, 0.1]`, `φ(s) = [1, 0.5, 2]`, `α = 0.1`, `δ = 0.4`

Compute new weights.

{{< pyrepl code="w = [0.5, -0.3, 0.1]\nphi = [1, 0.5, 2]\nalpha = 0.1\ndelta = 0.4   # TD error\n\n# TODO: w_new[i] = w[i] + alpha * delta * phi[i]\nw_new = None\nprint(w_new)   # [0.54, -0.28, 0.18]" height="220" >}}

{{< collapse summary="Solution" >}}
```python
w = [0.5, -0.3, 0.1]
phi = [1, 0.5, 2]
alpha = 0.1
delta = 0.4
w_new = [wi + alpha * delta * pi for wi, pi in zip(w, phi)]
print([round(x, 4) for x in w_new])   # [0.54, -0.28, 0.18]
```
{{< /collapse >}}

---

## B15 — Mini Q-learning agent

Implement tabular Q-learning on a 3×3 gridworld for 500 episodes. Report average steps to goal in last 100 evaluation episodes (greedy, ε=0).

{{< pyrepl code="import random\nrandom.seed(0)\n\n# 3x3 gridworld: start (0,0), goal (2,2)\n# reward: +1 at goal, -1 per step, -1 for wall\ndef step(state, action):\n    row, col = state\n    dr = [-1,1,0,0]; dc = [0,0,-1,1]\n    nr = row+dr[action]; nc = col+dc[action]\n    if not(0<=nr<=2 and 0<=nc<=2): return state,-1,False\n    if(nr,nc)==(2,2): return(2,2),1,True\n    return(nr,nc),-1,False\n\n# TODO: Q-learning for 500 episodes\n# Then evaluate for 100 episodes (epsilon=0), report mean steps\n\nQ = {}   # Q[(state,action)] = float\nalpha, gamma, epsilon = 0.1, 0.99, 0.1\n\n# Training\nfor ep in range(500):\n    pass   # TODO\n\n# Evaluation\nsteps_list = []\nfor _ in range(100):\n    pass   # TODO: greedy evaluation\n\nprint(f'Mean steps to goal: {sum(steps_list)/len(steps_list):.1f}')" height="360" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(0)

def step(state, action):
    row, col = state
    dr = [-1,1,0,0]; dc = [0,0,-1,1]
    nr = row+dr[action]; nc = col+dc[action]
    if not(0<=nr<=2 and 0<=nc<=2): return state,-1,False
    if(nr,nc)==(2,2): return(2,2),1,True
    return(nr,nc),-1,False

Q = {}
alpha, gamma, epsilon = 0.1, 0.99, 0.1
n_actions = 4

def get_q(s, a): return Q.get((s,a), 0.0)
def best_a(s): return max(range(n_actions), key=lambda a: get_q(s,a))

for ep in range(500):
    s = (0,0); done = False; steps = 0
    while not done and steps < 50:
        a = random.randrange(n_actions) if random.random()<epsilon else best_a(s)
        ns, r, done = step(s, a)
        target = r + (gamma * max(get_q(ns,a2) for a2 in range(n_actions)) if not done else 0)
        Q[(s,a)] = get_q(s,a) + alpha*(target - get_q(s,a))
        s = ns; steps += 1

steps_list = []
for _ in range(100):
    s=(0,0); done=False; steps=0
    while not done and steps < 50:
        s, _, done = step(s, best_a(s))
        steps += 1
    steps_list.append(steps)
print(f"Mean steps to goal: {sum(steps_list)/len(steps_list):.1f}")
```
{{< /collapse >}}

---

## Checklist

- [ ] B1–B7 (fundamentals): completed without hints on at least 5 of 7
- [ ] B8–B11 (math bridge): understood Bellman backup and TD update
- [ ] B12–B15 (RL programs): completed at least 2 of 4 without hints

If you can complete at least 10 of 15, you are ready for **[Volume 1: Mathematical Foundations](../curriculum/volume-01/)**.
