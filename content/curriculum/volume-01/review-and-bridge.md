---
title: "Volume 1 Review & Bridge to Volume 2"
description: "Review Volume 1 concepts and preview Volume 2. From dynamic programming (model-given) to model-free methods."
date: 2026-03-19T00:00:00Z
draft: false
weight: 100
tags: ["review", "bridge", "Volume 1", "Volume 2", "model-free", "dynamic programming"]
---

You have finished Volume 1. Before starting Volume 2, take this 10-minute review.

---

## Volume 1 Recap Quiz

{{< collapse summary="Q1. What are the three components of the RL framework?" >}}
**Agent** (learner), **Environment** (everything outside the agent), **Reward signal** (feedback). The agent observes states, takes actions, and receives rewards. Its goal is to maximize expected discounted return.
{{< /collapse >}}

{{< collapse summary="Q2. What assumption does Dynamic Programming require that Monte Carlo does not?" >}}
**A complete model of the environment**: transition probabilities P(s'|s,a) and reward function R(s,a,s'). Monte Carlo learns from actual episodes — no model needed.
{{< /collapse >}}

{{< collapse summary="Q3. What is the difference between policy evaluation and value iteration?" >}}
**Policy evaluation** computes V^π for a *given* policy π (uses Bellman expectation equation). **Value iteration** computes V* by applying the Bellman optimality operator — it directly finds the optimal policy without fixing a policy first.
{{< /collapse >}}

{{< collapse summary="Q4. Why can't we just run value iteration on a real robot?" >}}
Value iteration requires knowing P(s'|s,a) and R for all states. For a real robot: (1) the model may be unknown or too complex to specify; (2) the state space may be too large. We need model-free methods.
{{< /collapse >}}

{{< collapse summary="Q5. What is the main limitation of tabular Dynamic Programming?" >}}
It requires a table entry for every state (and every state-action pair for Q). For large state spaces (e.g. Atari: 10^{17000} possible screens), this is impossible. Volume 3+ addresses this with function approximation.
{{< /collapse >}}

---

## What Changes in Volume 2

| | Volume 1 (DP) | Volume 2 (Model-free) |
|---|---|---|
| **Model required?** | Yes | No |
| **How values are estimated** | Exact Bellman sweeps | Sampled episodes / transitions |
| **Convergence** | Exact (given model) | In expectation (with enough data) |
| **Key algorithms** | Policy eval, Policy iter, Value iter | Monte Carlo, TD(0), SARSA, Q-learning |
| **Bootstrapping** | Yes (full backup) | Monte Carlo: No. TD: Yes |

**The big insight:** Monte Carlo replaces the expectation over transitions with the *sample* return from one episode. TD methods go further — they bootstrap (use current estimates) so they can update after every step, not just at the end of an episode.

---

## Bridge Exercise

You implemented policy evaluation using the known transition model. Now imagine you **don't have** the model — you can only run episodes.

Modify the following to use sample episodes instead of Bellman sweeps:

{{< pyrepl code="import random\nrandom.seed(42)\n\n# The 3-state chain: A -> B -> C (terminal, reward=1)\n# We DON'T know the model. We only sample episodes.\n\ndef sample_episode(start='A'):\n    \"\"\"Return list of (state, reward) pairs.\"\"\"\n    traj = []\n    s = start\n    while s != 'C':\n        s_next = 'B' if s == 'A' else 'C'\n        r = 1 if s_next == 'C' else 0\n        traj.append((s, r))\n        s = s_next\n    return traj\n\n# TODO: run 1000 episodes starting from 'A', estimate V(A) and V(B)\n# V(s) = average of returns from first visit to s\nreturns = {'A': [], 'B': []}\nfor _ in range(1000):\n    ep = sample_episode('A')\n    # G = discounted_return (gamma=0.9)\n    # for each (state, reward) in ep, compute G from that point\n    pass\n\nfor s in ['A', 'B']:\n    if returns[s]:\n        print(f'V({s}) = {sum(returns[s])/len(returns[s]):.3f}')  # A~0.9, B~1.0" height="300" >}}

{{< collapse summary="Solution" >}}
```python
import random
random.seed(42)

def sample_episode(start='A'):
    traj = []
    s = start
    while s != 'C':
        s_next = 'B' if s == 'A' else 'C'
        r = 1 if s_next == 'C' else 0
        traj.append((s, r))
        s = s_next
    return traj

def discounted_return(rewards, gamma=0.9):
    return sum(gamma**t * r for t, r in enumerate(rewards))

returns = {'A': [], 'B': []}
visited = set()

for _ in range(1000):
    ep = sample_episode('A')
    visited.clear()
    states = [s for s, r in ep]
    rewards = [r for s, r in ep]
    for i, s in enumerate(states):
        if s in returns and s not in visited:
            visited.add(s)
            G = discounted_return(rewards[i:])
            returns[s].append(G)

for s in ['A', 'B']:
    if returns[s]:
        print(f"V({s}) = {sum(returns[s])/len(returns[s]):.3f}")
```
{{< /collapse >}}

**What changed:** Instead of computing V using transition probabilities, you sampled actual trajectories and averaged returns. That is Monte Carlo prediction — the first model-free method in Volume 2.

---

## Ready for Volume 2?

Before continuing, confirm:

- [ ] I can write the Bellman equation from memory.
- [ ] I understand why DP needs a model.
- [ ] I implemented policy evaluation (or followed the code closely).
- [ ] I understand the bridge exercise above — averaging sample returns to estimate V.

**Next:** [Volume 2: Tabular Methods & Classic Algorithms](../volume-02/)
