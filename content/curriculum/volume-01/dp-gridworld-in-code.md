---
title: "Dynamic Programming: Gridworld in Code"
description: "Code walkthrough for gridworld, iterative policy evaluation, and policy iteration."
date: 2026-03-10T00:00:00Z
weight: 9
draft: false
difficulty: 6
tags: ["dynamic programming", "gridworld", "code", "curriculum"]
keywords: ["policy evaluation", "policy iteration", "gridworld", "code"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Dp Gridworld In Code"
---

**Learning objectives**

- Implement a 4×4 gridworld environment (states, actions, transitions, rewards) in code.
- Implement iterative policy evaluation and stop when values converge.
- Implement policy iteration (evaluate then improve) and optionally value iteration.

## Gridworld in code

**States:** Use a 4×4 grid. States can be (row, col) or a flat index. Terminal states (0,0) and (3,3) have value 0 and are not updated.

**Actions:** 0=up, 1=down, 2=left, 3=right. Moving off the grid leaves the agent in place.

**Transitions:** Deterministic: from (r, c), “up” → (r-1, c) if r > 0 else (r, c). Same for down/left/right with bounds checks.

**Rewards:** -1 for each step in a non-terminal state. Terminal: no further reward (episode ends).

Example structure (Python):

```python
def step(s, a):
    r, c = s
    if a == 0: r = max(0, r - 1)
    elif a == 1: r = min(3, r + 1)
    elif a == 2: c = max(0, c - 1)
    else: c = min(3, c + 1)
    s_next = (r, c)
    done = s_next in [(0,0), (3,3)]
    reward = -1.0 if not done else 0.0
    return s_next, reward, done
```

(Adjust for your indexing and terminal handling.)

## Iterative policy evaluation in code

- Initialize \\(V(s) = 0\\) for all states.
- Loop: for each non-terminal state \\(s\\), compute \\(V_{new}(s) = \sum_a \pi(a|s) \bigl[ r(s,a) + \gamma V(s') \bigr]\\). For a uniform random policy, \\(\pi(a|s) = 0.25\\) for each action; \\(s'\\) and \\(r\\) come from your transition. Use **synchronous** updates: compute all \\(V_{new}\\) from the current \\(V\\), then set \\(V \leftarrow V_{new}\\).
- Stop when \\(\max_s |V_{new}(s) - V(s)| < \theta\\) (e.g. \\(\theta = 10^{-4}\\)).

## Policy iteration in code

- Start with an arbitrary policy (e.g. uniform random).
- **Evaluation:** Run iterative policy evaluation until convergence to get \\(V^\\pi\\).
- **Improvement:** For each non-terminal state \\(s\\), set \\(\pi(s) = \arg\max_a \bigl[ r(s,a) + \gamma V(s') \bigr]\\) (break ties arbitrarily).
- If the policy changed, go back to evaluation; else stop. The final policy is optimal for this MDP.

## Value iteration in code

Alternatively, use value iteration: repeatedly update \\(V(s) \leftarrow \max_a \bigl[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \bigr]\\) until convergence, then derive the greedy policy from \\(V\\). See [Chapter 9: Value Iteration](chapter-09/).

See [Gridworld](gridworld/) for the environment description, [Chapter 7: Policy Evaluation](chapter-07/) and [Chapter 8: Policy Iteration](chapter-08/) for theory, and [Windy Gridworld](windy-gridworld/) for the windy variant.
