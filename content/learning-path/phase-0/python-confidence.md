---
title: "Python Confidence Builder"
description: "25 mini-challenges combining variables, loops, functions, lists, dicts, and imports. Complete before Phase 1."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 1
weight: 2
tags: ["python", "phase 0", "practice", "exercises", "confidence builder"]
keywords: ["python exercises", "mini challenges", "RL programming", "beginner python", "confidence builder"]
roadmap_icon: "sparkles"
roadmap_color: "teal"
roadmap_phase_label: "Confidence Builder"
---

Work through these 25 challenges before moving to [Phase 1 (Math for RL)](../../math-for-rl/). Each challenge is small and self-contained. If you get stuck, use the hint; only check the solution after a genuine attempt.

Try each challenge in the interactive REPL below it, or in a `.py` file on your machine.

---

## Level 1 — Basics (Challenges 1–8)

### Challenge 1 — Print with variables

Set `name = "DQN"` and `version = 3`. Print: `"Algorithm: DQN version 3"` using an f-string.

{{< pyrepl code="name = 'DQN'\nversion = 3\n# TODO: print the line using an f-string\n" height="200" >}}

{{< collapse summary="Solution" >}}
```python
name = "DQN"
version = 3
print(f"Algorithm: {name} version {version}")
```
{{< /collapse >}}

---

### Challenge 2 — Type check

`gamma = "0.9"` is a string, not a float. Convert it to a float, square it, and print the result (expected: 0.81).

{{< pyrepl code="gamma = '0.9'\n# TODO: convert to float, square it\n" height="180" >}}

{{< collapse summary="Solution" >}}
```python
gamma = "0.9"
gamma = float(gamma)
print(gamma ** 2)   # 0.81
```
{{< /collapse >}}

---

### Challenge 3 — Conditional reward

Set `reward = -0.5`. Print `"Penalty"` if the reward is negative, `"Neutral"` if zero, `"Bonus"` if positive.

{{< pyrepl code="reward = -0.5\n# TODO: three-way if/elif/else\n" height="220" >}}

{{< collapse summary="Solution" >}}
```python
reward = -0.5
if reward < 0:
    print("Penalty")
elif reward == 0:
    print("Neutral")
else:
    print("Bonus")
```
{{< /collapse >}}

---

### Challenge 4 — Count steps

Use a `while` loop that starts with `step = 0` and increments until `step == 5`. Print each step. Then print `"Done"`.

{{< pyrepl code="step = 0\n# TODO: while loop\n" height="200" >}}

{{< collapse summary="Solution" >}}
```python
step = 0
while step < 5:
    print("Step", step)
    step += 1
print("Done")
```
{{< /collapse >}}

---

### Challenge 5 — Sum rewards

Use a `for` loop to compute the sum of `rewards = [0.5, 0.3, -0.2, 1.0, 0.0]`. Do **not** use `sum()`. Print the result (expected: 1.6).

{{< pyrepl code="rewards = [0.5, 0.3, -0.2, 1.0, 0.0]\ntotal = 0\n# TODO: for loop\nprint(total)" height="200" >}}

{{< collapse summary="Solution" >}}
```python
rewards = [0.5, 0.3, -0.2, 1.0, 0.0]
total = 0
for r in rewards:
    total += r
print(f"Total: {total}")   # 1.6
```
{{< /collapse >}}

---

### Challenge 6 — Function: clamp

Write `clamp(x, lo, hi)` that returns `lo` if `x < lo`, `hi` if `x > hi`, else `x`. (Used in PPO to clip ratios.) Test with `clamp(1.5, 0.8, 1.2)` → `1.2`.

{{< pyrepl code="def clamp(x, lo, hi):\n    pass  # TODO\n\nprint(clamp(1.5, 0.8, 1.2))  # 1.2\nprint(clamp(0.5, 0.8, 1.2))  # 0.8\nprint(clamp(1.0, 0.8, 1.2))  # 1.0" height="240" >}}

{{< collapse summary="Solution" >}}
```python
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

print(clamp(1.5, 0.8, 1.2))   # 1.2
print(clamp(0.5, 0.8, 1.2))   # 0.8
print(clamp(1.0, 0.8, 1.2))   # 1.0
```
{{< /collapse >}}

---

### Challenge 7 — Build a trajectory list

Create an empty list `trajectory`. Append 5 tuples `(step, step * 2, step * 0.1)` representing `(step, state, reward)` for steps 0–4. Print the list.

{{< pyrepl code="trajectory = []\n# TODO: loop and append tuples\nprint(trajectory)" height="200" >}}

{{< collapse summary="Solution" >}}
```python
trajectory = []
for step in range(5):
    trajectory.append((step, step * 2, step * 0.1))
print(trajectory)
```
{{< /collapse >}}

---

### Challenge 8 — Dict lookup with default

Create `Q = {("s0", "up"): 0.5, ("s0", "down"): -0.2}`. Write a function `get_q(Q, state, action)` that returns the Q-value if it exists, else `0.0`. Test with a missing key like `("s1", "up")`.

{{< pyrepl code="Q = {('s0','up'): 0.5, ('s0','down'): -0.2}\n\ndef get_q(Q, state, action):\n    pass  # TODO: use .get()\n\nprint(get_q(Q, 's0', 'up'))    # 0.5\nprint(get_q(Q, 's1', 'up'))    # 0.0" height="240" >}}

{{< collapse summary="Solution" >}}
```python
Q = {("s0","up"): 0.5, ("s0","down"): -0.2}

def get_q(Q, state, action):
    return Q.get((state, action), 0.0)

print(get_q(Q, "s0", "up"))    # 0.5
print(get_q(Q, "s1", "up"))    # 0.0
```
{{< /collapse >}}

---

## Level 2 — Combining skills (Challenges 9–17)

### Challenge 9 — Coin flip fraction

Import `random`. Simulate flipping a fair coin 1000 times (`random.choice([0, 1])`). Print the fraction of heads (expected ≈ 0.5). Set `random.seed(42)` first.

{{< pyrepl code="import random\nrandom.seed(42)\n# TODO: simulate 1000 flips, count heads\n" height="200" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)
n = 1000
heads = sum(random.choice([0, 1]) for _ in range(n))
print(f"Fraction of heads: {heads / n:.3f}")   # ≈ 0.5
```
{{< /collapse >}}

---

### Challenge 10 — Die roll counts

Roll a 6-sided die 600 times. Store counts in a dict `{face: count}`. Print each face and its count (expected ≈ 100 each).

{{< pyrepl code="import random\nrandom.seed(0)\ncounts = {}\n# TODO: roll 600 times, update counts dict\nfor face, count in sorted(counts.items()):\n    print(f'Face {face}: {count}')" height="240" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(0)
counts = {}
for _ in range(600):
    face = random.randint(1, 6)
    counts[face] = counts.get(face, 0) + 1
for face, count in sorted(counts.items()):
    print(f"Face {face}: {count}")
```
{{< /collapse >}}

---

### Challenge 11 — Discounted return

Write `discounted_return(rewards, gamma=0.9)` without using NumPy. Test:
- `[0, 0, 1]` → `0.81`
- `[1, 0, 0]` → `1.0`
- `[1, 1, 1]` → ≈`2.71`

{{< pyrepl code="def discounted_return(rewards, gamma=0.9):\n    pass  # TODO\n\nprint(discounted_return([0, 0, 1]))   # 0.81\nprint(discounted_return([1, 0, 0]))   # 1.0\nprint(discounted_return([1, 1, 1]))   # ~2.71" height="240" >}}

{{< collapse summary="Solution" >}}
```python
def discounted_return(rewards, gamma=0.9):
    return sum(gamma**t * r for t, r in enumerate(rewards))

print(discounted_return([0, 0, 1]))    # 0.81
print(discounted_return([1, 0, 0]))    # 1.0
print(f"{discounted_return([1,1,1]):.4f}")  # 2.7100
```
{{< /collapse >}}

---

### Challenge 12 — Find the max Q-action

Given `Q_s = {"up": 0.3, "down": -0.1, "left": 0.7, "right": 0.2}`, write `best_action(Q_s)` that returns the key with the highest value. Expected: `"left"`.

{{< pyrepl code="Q_s = {'up': 0.3, 'down': -0.1, 'left': 0.7, 'right': 0.2}\n\ndef best_action(Q_s):\n    pass  # TODO: max(Q_s, key=Q_s.get)\n\nprint(best_action(Q_s))   # 'left'" height="220" >}}

{{< collapse summary="Solution" >}}
```python
Q_s = {"up": 0.3, "down": -0.1, "left": 0.7, "right": 0.2}

def best_action(Q_s):
    return max(Q_s, key=Q_s.get)

print(best_action(Q_s))   # left
```
{{< /collapse >}}

---

### Challenge 13 — Valid grid moves

Write `valid_moves(row, col, n=4)` that returns a list of valid actions (0=up, 1=down, 2=left, 3=right) for a cell in an n×n grid (actions that stay in bounds).

{{< pyrepl code="def valid_moves(row, col, n=4):\n    pass  # TODO\n\nprint(valid_moves(0, 0))   # [1, 3] (down, right only)\nprint(valid_moves(2, 2))   # [0, 1, 2, 3]\nprint(valid_moves(3, 3))   # [0, 2] (up, left only)" height="240" >}}

{{< collapse summary="Solution" >}}
```python
def valid_moves(row, col, n=4):
    deltas = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
    valid = []
    for action, (dr, dc) in deltas.items():
        nr, nc = row+dr, col+dc
        if 0 <= nr < n and 0 <= nc < n:
            valid.append(action)
    return valid

print(valid_moves(0, 0))   # [1, 3]
print(valid_moves(2, 2))   # [0, 1, 2, 3]
print(valid_moves(3, 3))   # [0, 2]
```
{{< /collapse >}}

---

### Challenge 14 — Episode simulator

Write `simulate_episode(max_steps=10)` that:
- Starts at position 0
- Each step: move +1 with 70% prob, -1 with 30% prob
- Stop if position ≥ 5 or ≤ -5, or after `max_steps`
- Return the list of positions visited

{{< pyrepl code="import random\nrandom.seed(7)\n\ndef simulate_episode(max_steps=10):\n    pass  # TODO\n\npositions = simulate_episode()\nprint(positions)" height="240" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(7)

def simulate_episode(max_steps=10):
    pos = 0
    positions = [pos]
    for _ in range(max_steps):
        pos += 1 if random.random() < 0.7 else -1
        positions.append(pos)
        if abs(pos) >= 5:
            break
    return positions

print(simulate_episode())
```
{{< /collapse >}}

---

### Challenge 15 — Running mean

Write `running_mean(values)` that returns a list where the i-th element is the average of `values[0:i+1]` (used for smoothing reward curves).

{{< pyrepl code="def running_mean(values):\n    pass  # TODO\n\nprint(running_mean([1, 3, 2, 4]))   # [1.0, 2.0, 2.0, 2.5]" height="200" >}}

{{< collapse summary="Solution" >}}
```python
def running_mean(values):
    result = []
    total = 0
    for i, v in enumerate(values):
        total += v
        result.append(total / (i + 1))
    return result

print(running_mean([1, 3, 2, 4]))   # [1.0, 2.0, 2.0, 2.5]
```
{{< /collapse >}}

---

### Challenge 16 — Epsilon-greedy

Write `epsilon_greedy(q_values, epsilon=0.1)` that returns a random action index with probability `epsilon`, else the greedy action index.

{{< pyrepl code="import random\nrandom.seed(1)\n\ndef epsilon_greedy(q_values, epsilon=0.1):\n    pass  # TODO\n\n# With epsilon=0, should always return argmax\nq = [0.2, 0.7, 0.1]\nprint([epsilon_greedy(q, 0.0) for _ in range(5)])   # [1,1,1,1,1]" height="240" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(1)

def epsilon_greedy(q_values, epsilon=0.1):
    if random.random() < epsilon:
        return random.randrange(len(q_values))
    return q_values.index(max(q_values))

q = [0.2, 0.7, 0.1]
print([epsilon_greedy(q, 0.0) for _ in range(5)])   # [1,1,1,1,1]
```
{{< /collapse >}}

---

### Challenge 17 — Count action visits

Simulate 200 steps of epsilon-greedy with `q = [0.2, 0.7, 0.1, 0.4]` and `epsilon=0.2`. Count how many times each action was chosen. Print the counts as a dict.

{{< pyrepl code="import random\nrandom.seed(42)\nq = [0.2, 0.7, 0.1, 0.4]\n# TODO: 200 steps, epsilon=0.2, count each action\ncounts = {}\nprint(counts)" height="220" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)
q = [0.2, 0.7, 0.1, 0.4]
counts = {i: 0 for i in range(len(q))}
for _ in range(200):
    action = epsilon_greedy(q, 0.2)
    counts[action] += 1
print(counts)
```
{{< /collapse >}}

---

## Level 3 — Small RL programs (Challenges 18–25)

### Challenge 18 — Incremental mean

Write `incremental_mean_update(Q_old, n, reward)` that returns the new mean after seeing `reward` as the n-th observation: `Q_new = Q_old + (reward - Q_old) / n`. Test: start with Q=0, update with rewards [1.2, 0.8, 1.0, 1.4]. Final mean should be 1.1.

{{< pyrepl code="def incremental_mean_update(Q_old, n, reward):\n    pass  # TODO\n\nQ = 0.0\nfor n, r in enumerate([1.2, 0.8, 1.0, 1.4], 1):\n    Q = incremental_mean_update(Q, n, r)\nprint(f'Final mean: {Q:.2f}')  # 1.10" height="240" >}}

{{< collapse summary="Solution" >}}
```python
def incremental_mean_update(Q_old, n, reward):
    return Q_old + (reward - Q_old) / n

Q = 0.0
for n, r in enumerate([1.2, 0.8, 1.0, 1.4], 1):
    Q = incremental_mean_update(Q, n, r)
print(f"Final mean: {Q:.2f}")   # 1.10
```
{{< /collapse >}}

---

### Challenge 19 — 3-armed bandit simulation

Simulate a 3-armed bandit for 300 steps with true means `[0.3, 0.7, 0.1]`. Use epsilon-greedy (ε=0.1) and incremental mean updates. Print the final Q estimates and which arm was pulled most.

{{< pyrepl code="import random\nrandom.seed(42)\n\ntrue_means = [0.3, 0.7, 0.1]\nk = len(true_means)\nQ = [0.0] * k\nN = [0] * k\n\n# TODO: 300 steps of epsilon-greedy + incremental update\n\nprint('Q estimates:', [round(q, 3) for q in Q])\nprint('Pull counts:', N)" height="280" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)

true_means = [0.3, 0.7, 0.1]
k = len(true_means)
Q = [0.0] * k
N = [0] * k

epsilon = 0.1
for step in range(300):
    if random.random() < epsilon:
        action = random.randrange(k)
    else:
        action = Q.index(max(Q))
    reward = random.gauss(true_means[action], 1)
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]

print("Q estimates:", [round(q, 3) for q in Q])
print("Pull counts:", N)
print("Most pulled:", Q.index(max(Q)))
```
{{< /collapse >}}

---

### Challenge 20 — Random walk value estimation

Simulate a 1D random walk: states 0–6, start at 3, walk left/right equally. Episode ends at 0 or 6 (reward +1 at 6, 0 at 0). Run 2000 episodes. Estimate V(s) = fraction of episodes from s that reached 6.

{{< pyrepl code="import random\nrandom.seed(0)\n\ndef random_walk_episode(start=3):\n    # Returns list of (state, reward) until terminal\n    pass\n\n# TODO: run 2000 episodes, estimate V(s) for s=1..5\n" height="280" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(0)

def random_walk_episode(start=3):
    s = start
    traj = []
    while s not in (0, 6):
        s += 1 if random.random() < 0.5 else -1
        r = 1 if s == 6 else 0
        traj.append((s, r))
    return traj

returns = {s: [] for s in range(1, 6)}
for _ in range(2000):
    ep = random_walk_episode()
    for i, (s, r) in enumerate(ep):
        if s in returns:
            G = sum(rr for _, rr in ep[i:])
            returns[s].append(G)

for s in range(1, 6):
    v = sum(returns[s]) / len(returns[s]) if returns[s] else 0
    print(f"V({s}) = {v:.3f}  (true = {s/6:.3f})")
```
{{< /collapse >}}

---

### Challenge 21 — Find the bug (1)

This function is supposed to compute discounted return but has a bug. Find and fix it.

```python
def broken_return(rewards, gamma=0.9):
    G = 0
    for t, r in enumerate(rewards):
        G += r   # bug: missing discount
    return G
```

{{< pyrepl code="def broken_return(rewards, gamma=0.9):\n    G = 0\n    for t, r in enumerate(rewards):\n        G += r   # bug: missing discount\n    return G\n\n# Expected: 0.81, got: 1\nprint(broken_return([0, 0, 1]))   # should be 0.81\n# Fix it below:\ndef fixed_return(rewards, gamma=0.9):\n    pass\nprint(fixed_return([0, 0, 1]))   # 0.81" height="280" >}}

{{< collapse summary="Solution" >}}
```python
def fixed_return(rewards, gamma=0.9):
    G = 0
    for t, r in enumerate(rewards):
        G += gamma**t * r   # apply discount
    return G

print(fixed_return([0, 0, 1]))    # 0.81
print(fixed_return([1, 1, 1]))    # 2.71
```
{{< /collapse >}}

---

### Challenge 22 — Find the bug (2)

This epsilon-greedy function explores when it should exploit and vice versa. Fix it.

```python
def bad_epsilon_greedy(Q, epsilon=0.1):
    import random
    if random.random() > epsilon:  # bug: should be <
        return random.randrange(len(Q))
    return Q.index(max(Q))
```

{{< pyrepl code="import random\nrandom.seed(5)\n\ndef bad_epsilon_greedy(Q, epsilon=0.1):\n    if random.random() > epsilon:  # bug\n        return random.randrange(len(Q))\n    return Q.index(max(Q))\n\n# With epsilon=0.0, greedy should ALWAYS return 1\nQ = [0.1, 0.9, 0.3]\nprint([bad_epsilon_greedy(Q, 0.0) for _ in range(5)])\n# This should be [1,1,1,1,1] but isn't — find and fix the bug\n\ndef fixed_eg(Q, epsilon=0.1):\n    pass\n" height="280" >}}

{{< collapse summary="Solution" >}}
```python
import random

def fixed_eg(Q, epsilon=0.1):
    if random.random() < epsilon:   # < not >
        return random.randrange(len(Q))
    return Q.index(max(Q))

random.seed(5)
Q = [0.1, 0.9, 0.3]
print([fixed_eg(Q, 0.0) for _ in range(5)])   # [1,1,1,1,1]
```
{{< /collapse >}}

---

### Challenge 23 — Q-table update

Implement one step of Q-learning: `Q[s][a] += alpha * (r + gamma * max(Q[s_next]) - Q[s][a])`.

Given:
- `Q = {"A": [0.0, 0.5], "B": [0.3, 0.2]}`
- Transition: `s="A"`, `a=0`, `r=1`, `s_next="B"`, `done=False`
- `alpha=0.1`, `gamma=0.9`

Print the updated `Q["A"][0]`.

{{< pyrepl code="Q = {'A': [0.0, 0.5], 'B': [0.3, 0.2]}\nalpha, gamma = 0.1, 0.9\ns, a, r, s_next, done = 'A', 0, 1, 'B', False\n\n# TODO: Q-learning update\n# target = r + gamma * max(Q[s_next]) if not done else r\n# Q[s][a] += alpha * (target - Q[s][a])\n\nprint(f'Q[A][0] = {Q[\"A\"][0]:.4f}')   # expected: 0.127" height="260" >}}

{{< collapse summary="Solution" >}}
```python
Q = {"A": [0.0, 0.5], "B": [0.3, 0.2]}
alpha, gamma = 0.1, 0.9
s, a, r, s_next, done = "A", 0, 1, "B", False

target = r + gamma * max(Q[s_next]) if not done else r
Q[s][a] += alpha * (target - Q[s][a])
print(f"Q[A][0] = {Q['A'][0]:.4f}")   # 0.1 * (1 + 0.9*0.3 - 0) = 0.127
```
{{< /collapse >}}

---

### Challenge 24 — Multi-episode return tracking

Run 50 episodes of a random agent on a 3×3 gridworld (use your `step()` function from Phase 0). Each episode: start at (0,0), take random actions, stop at (2,2) or after 20 steps. Collect the total (undiscounted) reward per episode. Print the mean and max episode return.

{{< pyrepl code="import random\nrandom.seed(0)\n\n# Use the step function from Chapter 1\ndef step(state, action):\n    row, col = state\n    moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}\n    dr,dc = moves[action]\n    nr,nc = row+dr,col+dc\n    if not(0<=nr<=2 and 0<=nc<=2): return state,-1,False\n    if(nr,nc)==(2,2): return(2,2),1,True\n    return(nr,nc),0,False\n\n# TODO: run 50 episodes, collect total reward per episode\nepisode_returns = []\n\nprint(f'Mean return: {sum(episode_returns)/len(episode_returns):.2f}')\nprint(f'Max return:  {max(episode_returns)}')" height="300" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(0)

def step(state, action):
    row, col = state
    moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
    dr,dc = moves[action]
    nr,nc = row+dr,col+dc
    if not(0<=nr<=2 and 0<=nc<=2): return state,-1,False
    if(nr,nc)==(2,2): return(2,2),1,True
    return(nr,nc),0,False

episode_returns = []
for _ in range(50):
    state, G, done = (0,0), 0, False
    for _ in range(20):
        if done: break
        a = random.randint(0, 3)
        state, r, done = step(state, a)
        G += r
    episode_returns.append(G)

print(f"Mean return: {sum(episode_returns)/len(episode_returns):.2f}")
print(f"Max return:  {max(episode_returns)}")
```
{{< /collapse >}}

---

### Challenge 25 — Full mini-agent

Implement a complete bandit agent: 5 arms, true means drawn from Normal(0, 1), run 500 steps. Use epsilon-greedy (ε=0.1) and incremental updates. Report: (1) estimated Q for each arm, (2) which arm the agent converged to, (3) whether it is the true best arm.

{{< pyrepl code="import random\nrandom.seed(99)\n\nk = 5\ntrue_means = [random.gauss(0, 1) for _ in range(k)]\nQ = [0.0] * k\nN = [0] * k\nepsilon = 0.1\n\n# TODO: 500 steps\n\nprint('True means:', [round(m,3) for m in true_means])\nprint('Q estimates:', [round(q,3) for q in Q])\nprint('Pull counts:', N)\nprint('Best arm (true):', true_means.index(max(true_means)))\nprint('Best arm (estimated):', Q.index(max(Q)))" height="320" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(99)

k = 5
true_means = [random.gauss(0, 1) for _ in range(k)]
Q = [0.0] * k
N = [0] * k
epsilon = 0.1

for _ in range(500):
    if random.random() < epsilon:
        a = random.randrange(k)
    else:
        a = Q.index(max(Q))
    r = random.gauss(true_means[a], 1)
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]

print("True means:    ", [round(m,3) for m in true_means])
print("Q estimates:   ", [round(q,3) for q in Q])
print("Pull counts:   ", N)
print("Best (true):   ", true_means.index(max(true_means)))
print("Best (learned):", Q.index(max(Q)))
```
{{< /collapse >}}

---

## Checklist

Before moving to Phase 1, confirm:

- [ ] I completed at least 20 of the 25 challenges.
- [ ] I understood why each solution works (not just copied it).
- [ ] I can write `discounted_return`, `epsilon_greedy`, and `incremental_mean_update` from memory.
- [ ] I can create and query a Q-table as a Python dict.

When all four are checked, proceed to **[Phase 1 — Math for RL](../../math-for-rl/)**.
