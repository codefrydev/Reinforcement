---
title: "Volume 1 Drills — Mathematical Foundations"
description: "15 short drill problems for Volume 1: discounted return, MDPs, value functions, Bellman equations, and dynamic programming."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 6
weight: 99
tags: ["drills", "volume 1", "MDP", "Bellman", "dynamic programming", "practice"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Drills"
---

{{< notebook path="volume-01/vol01_drills.ipynb" title="Open drills notebook (interactive)" >}}

Short problems for Volume 1. Aim for under 5 minutes per problem. All solutions are in collapsible sections.

---

## Recall (R) — State definitions and rules

**R1.** Write the Bellman expectation equation for V^π(s) in words: "The value of state s under policy π is \_\_\_."

{{< collapse summary="Answer" >}}
"The value of state s under policy π is the **expected immediate reward plus the discounted expected value of the next state**, averaged over actions (weighted by the policy) and transitions."

Formally: V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')].
{{< /collapse >}}

---

**R2.** What are the five components of an MDP? Write the tuple notation.

{{< collapse summary="Answer" >}}
(S, A, P, R, γ): **S** = state space, **A** = action space, **P** = transition probabilities P(s'|s,a), **R** = reward function R(s,a,s'), **γ** = discount factor.
{{< /collapse >}}

---

**R3.** What is the Markov property? Why does RL need it?

{{< collapse summary="Answer" >}}
The **Markov property**: P(S_{t+1}|S_0,...,S_t, A_0,...,A_t) = P(S_{t+1}|S_t, A_t). The future depends only on the current state and action, not the full history.

**Why RL needs it:** Without the Markov property, the agent would need to remember its entire history to make good decisions. The Markov property allows decision-making based only on the current state — making the problem tractable.
{{< /collapse >}}

---

**R4.** What is the difference between policy evaluation and policy iteration?

{{< collapse summary="Answer" >}}
**Policy evaluation:** compute V^π for a fixed policy π (using Bellman expectation equations iteratively).

**Policy iteration:** alternate between policy evaluation and policy improvement (act greedily w.r.t. V^π) until the policy stops changing. Converges to the optimal policy π*.
{{< /collapse >}}

---

**R5.** When does value iteration converge, and how do you know?

{{< collapse summary="Answer" >}}
Value iteration converges when the maximum change in V across all states in one sweep is less than a threshold ε (e.g. 1e-6): max_s |V_new(s) - V_old(s)| < ε.

The Bellman optimality operator is a contraction mapping (with factor γ), guaranteeing convergence to V*.
{{< /collapse >}}

---

## Compute (C) — Numerical exercises

**C1.** Compute the discounted return for rewards = [2, 0, -1] and γ = 0.9.

{{< pyrepl code="rewards = [2, 0, -1]\ngamma = 0.9\nG = sum(gamma**t * r for t, r in enumerate(rewards))\nprint(f'G = {G:.4f}')  # should be 2 + 0 + 0.81*(-1)" height="180" >}}

{{< collapse summary="Answer" >}}
G = 2×1 + 0×0.9 + (-1)×0.81 = 2 - 0.81 = **1.19**.
{{< /collapse >}}

---

**C2.** In a 2-state MDP (A and B), both non-terminal: from A, go to B with probability 1, reward = 0. From B, go to A with probability 1, reward = 1. γ = 0.9. Current estimates: V(A) = 0.5, V(B) = 0.4.

Compute one Bellman backup: V_new(A) and V_new(B).

{{< pyrepl code="V_A, V_B = 0.5, 0.4\ngamma = 0.9\n# V_new(A) = 0 + gamma * V_B\n# V_new(B) = 1 + gamma * V_A\nV_new_A = 0 + gamma * V_B\nV_new_B = 1 + gamma * V_A\nprint(f'V_new(A) = {V_new_A:.4f}')   # 0.36\nprint(f'V_new(B) = {V_new_B:.4f}')   # 1.45" height="220" >}}

{{< collapse summary="Answer" >}}
V_new(A) = 0 + 0.9 × 0.4 = **0.36**. V_new(B) = 1 + 0.9 × 0.5 = **1.45**.
{{< /collapse >}}

---

**C3.** A 3×3 gridworld. State (2,2) is the goal (reward +1). All other states give reward 0. Discount γ=0.99. After many iterations, V*(2,2) = 1/(1-γ) if the agent stays in goal, or 1 if the episode ends. What is V*(2,1) if the agent is one step from the goal and γ=0.99?

{{< collapse summary="Answer" >}}
If the episode terminates at the goal: V*(2,1) = 0 + γ × V*(2,2) where V*(goal at terminal) = 1 (one-step reward). So V*(2,1) = 0.99 × 1 = **0.99**.

If the agent stays in the goal state with reward +1 per step: V*(goal) = 1/(1-0.99) = 100, and V*(2,1) = 0 + 0.99 × 100 = 99. (Depends on episodic vs continuing formulation — episodic gives the simpler answer.)
{{< /collapse >}}

---

**C4.** Policy iteration step: given V^π(A)=0.3, V^π(B)=0.8, and transitions: from state S, action "go to A" gives R=0, action "go to B" gives R=0.1. γ=0.9. What is the greedy action?

{{< collapse summary="Answer" >}}
Q(S, go-to-A) = 0 + 0.9 × 0.3 = 0.27.
Q(S, go-to-B) = 0.1 + 0.9 × 0.8 = 0.1 + 0.72 = 0.82.

Greedy action: **go to B** (Q=0.82 > Q=0.27).
{{< /collapse >}}

---

**C5.** One step of value iteration for a 2-action state: actions up and down, both deterministic.
- up → reward 0, next state has V=0.5
- down → reward 1, next state has V=0.2
- γ=0.9

Compute V*(s).

{{< pyrepl code="gamma = 0.9\nq_up = 0 + gamma * 0.5\nq_down = 1 + gamma * 0.2\nV_star = max(q_up, q_down)\nprint(f'Q(up) = {q_up}, Q(down) = {q_down}')\nprint(f'V*(s) = {V_star}')" height="200" >}}

{{< collapse summary="Answer" >}}
Q(up) = 0 + 0.9×0.5 = 0.45. Q(down) = 1 + 0.9×0.2 = 1.18. V*(s) = max(0.45, 1.18) = **1.18** (optimal action: down).
{{< /collapse >}}

---

## Code (K) — Implementation

**K1.** Implement `discounted_return(rewards, gamma)` without using NumPy.

{{< pyrepl code="def discounted_return(rewards, gamma=0.9):\n    # TODO\n    pass\n\nprint(discounted_return([0, 0, 1], 0.9))   # 0.81\nprint(discounted_return([1, 1, 1], 0.9))   # 2.71" height="200" >}}

{{< collapse summary="Solution" >}}
```python
def discounted_return(rewards, gamma=0.9):
    return sum(gamma**t * r for t, r in enumerate(rewards))
```
{{< /collapse >}}

---

**K2.** Implement one sweep of iterative policy evaluation for a linear chain of 3 states (A, B, C) where C is terminal. Policy: always go right. Reward: 0 for A→B, 0 for B→C, +1 reaching C. γ=0.9.

{{< pyrepl code="V = {'A': 0.0, 'B': 0.0, 'C': 1.0}   # C is terminal/goal\ngamma = 0.9\n\n# One sweep (right = deterministic)\n# V_new(A) = 0 + gamma * V(B)\n# V_new(B) = 0 + gamma * V(C)\n# V(C) stays 1 (terminal)\n\nfor _ in range(20):\n    V_new = {}\n    V_new['A'] = 0 + gamma * V['B']\n    V_new['B'] = 0 + gamma * V['C']\n    V_new['C'] = 1.0\n    V = V_new\n\nprint({s: round(v, 4) for s, v in V.items()})" height="260" >}}

{{< collapse summary="Solution" >}}
After 20 iterations: V(C)=1.0, V(B)=0.9, V(A)=0.81 (geometric series under γ=0.9).
{{< /collapse >}}

---

## Debug (D) — Find and fix the bug

**D1.** This value iteration code has a bug. Find and fix it.

```python
def value_iteration(V, transitions, gamma=0.9, n_iter=100):
    for _ in range(n_iter):
        for state in V:
            # transitions[state] = list of (reward, next_state)
            V[state] = max(r + gamma * V[next_s]
                          for r, next_s in transitions[state])
    return V

# Bug: V is mutated during iteration — next_s may use updated values
```

{{< pyrepl code="# The bug: V is updated in place, so later states see\n# already-updated values from the same sweep.\n# Fix: use a separate V_new dict, then update V at end of sweep.\n\ndef value_iteration_fixed(V, transitions, gamma=0.9, n_iter=100):\n    V = dict(V)   # copy\n    for _ in range(n_iter):\n        V_new = {}\n        for state in V:\n            V_new[state] = max(r + gamma * V[next_s]\n                              for r, next_s in transitions[state])\n        V = V_new\n    return V\n\n# Test on 2-state example\nV0 = {'A': 0.0, 'B': 0.0}\ntrans = {'A': [(0, 'B')], 'B': [(1, 'A')]}\nprint(value_iteration_fixed(V0, trans, n_iter=200))" height="300" >}}

{{< collapse summary="Answer" >}}
The bug: in-place mutation of V during a sweep means earlier states (already updated) are used when computing later states, giving inconsistent results. The fix: compute all new values using the old V, then replace V.
{{< /collapse >}}

---

**D2.** Find the bug in this epsilon-greedy implementation:

```python
def epsilon_greedy(Q, epsilon=0.1):
    import random
    if random.random() > epsilon:    # Bug!
        return random.randrange(len(Q))
    return Q.index(max(Q))
```

{{< collapse summary="Answer" >}}
The condition is **reversed**. `random.random() > epsilon` is True ~90% of the time (with ε=0.1), so the code mostly does random exploration (not greedy). Fix: `if random.random() < epsilon:` for exploration, `else: return Q.index(max(Q))` for exploitation.
{{< /collapse >}}

---

## Challenge (X)

**X1.** Implement full policy iteration on a 3×3 gridworld: start at (0,0), goal at (2,2) with reward +1, -1 per step, walls give -1. Use a tabular representation. Report the optimal policy (one arrow per cell) and number of iterations until convergence.

{{< pyrepl code="import numpy as np\n# 3x3 gridworld: states=(row,col), actions=0..3\n# Implement policy iteration here\n# Start: random policy\n# Repeat: policy eval + policy improvement\n# Stop when policy doesn't change\nprint('Implement policy iteration...')" height="200" >}}

{{< collapse summary="Hint" >}}
1. Initialize V=0, π=random for all states.
2. Policy evaluation: iterate Bellman expectation until convergence.
3. Policy improvement: for each state, set π(s) = argmax_a Q(s,a) using current V.
4. Repeat until π doesn't change.
Expected result: arrows pointing toward (2,2) along the optimal path.
{{< /collapse >}}
