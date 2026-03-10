---
title: "Chapter 10: Limitations of Dynamic Programming"
description: "State and transition count for 10×10 gridworld; function approximation."
date: 2026-03-10T00:00:00Z
weight: 10
draft: false
tags: ["dynamic programming", "function approximation", "tabular limits", "curriculum"]
keywords: ["limitations of DP", "function approximation", "state count", "tabular methods"]
---

**Learning objectives**

- Compute the number of states and transition probabilities for a small finite MDP.
- Explain why tabular methods (storing a value per state or state-action) do not scale.
- Describe how function approximation (e.g. linear or neural) generalizes across states.

**Concept and real-world RL**

**Dynamic programming** (policy iteration, value iteration) assumes we can store a value for every state (or state-action) and iterate over all of them. In a 10×10 grid that is 100 states—manageable. In real problems (continuous state spaces, or discrete but huge spaces like board games or high-dimensional sensors), the number of states is enormous or infinite, so we cannot store a table. **Function approximation** uses a parameterized function (e.g. \\(V(s; \\theta)\\) or \\(Q(s,a; \\theta)\\)) so that a fixed number of parameters \\(\\theta\\) represent values for *all* states; we learn \\(\\theta\\) from data. This is the bridge to deep RL (DQN, policy gradients) in Volumes 3–5.

**Exercise:** Consider a 10×10 gridworld with 4 actions. How many states and how many transition probabilities would you need to store explicitly? Discuss why this becomes infeasible for larger grids and how function approximation addresses it.

**Professor's hints**

- **States:** 10×10 = 100 states (if we do not include "terminal" as extra states). If terminals are separate, adjust accordingly.
- **Transition probabilities:** For each (state, action) we need a distribution over (next state, reward). So at least 100 × 4 = 400 (s,a) pairs. For each (s,a), you might store \\(|S| \times |R|\\) numbers (next state and reward combinations). A simple model: each (s,a) leads to one of at most 4 next states (neighbors) plus reward; so roughly 400 × (a few) entries. More generally, \\(|S| \\times |A| \\times (|S'| \\times |R|)\\) or similar—the key is it grows with \\(|S|^2\\) and \\(|A|\\).
- **Infeasibility:** For a 100×100 grid, 10,000 states; for continuous state (e.g. robot joint angles), infinitely many. Storage and computation become impossible.
- **Function approximation:** Instead of \\(V(s)\\) for each \\(s\\), use \\(V(s; w) = w^T \\phi(s)\\) (linear) or a neural network. We store only \\(w\\) (or network weights); \\(\\phi(s)\\) can be features (e.g. coordinates, tile coding). Learning updates \\(w\\) from samples so that \\(V(s; w)\\) approximates the true value for *visited* states and *generalizes* to unseen states.

**Common pitfalls**

- **Underestimating growth:** Doubling grid side length quadruples the number of states. In high dimensions (e.g. 10 dimensions, 10 cells each), state count is \\(10^{10}\\)—no tabular method can handle that.
- **Confusing transition count with value count:** We need both: transition model \\(P(s',r|s,a)\\) for DP, and we store \\(V(s)\\) or \\(Q(s,a)\\). The transition model is often the bigger storage (especially if we do not have a compact model).
- **Assuming FA is always better:** Function approximation can generalize wrongly (e.g. bad values in unseen states) and can diverge if not carefully designed. Tabular methods are stable when the state space is small enough.

{{< collapse summary="Worked solution (warm-up: tabular Q size)" >}}
**Warm-up:** A 5×5 grid has 25 states. How many entries does a tabular \\(Q(s,a)\\) table have if there are 4 actions?

**Step 1:** We need one \\(Q(s,a)\\) value for each state \\(s\\) and each action \\(a\\). So number of entries = \\(|S| \times |A| = 25 \times 4 = 100\\).

**Explanation:** Tabular \\(Q\\) grows linearly with states and actions. For a 10×10 grid with 4 actions that is 400 entries; for huge or continuous state spaces we cannot store a table, so we use function approximation (e.g. \\(Q(s,a; \\theta)\\) with a fixed number of parameters \\(\\theta\\)) to generalize across states.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** A 5×5 grid has 25 states. How many entries does a tabular \\(Q(s,a)\\) table have if there are 4 actions? (Answer: 25 × 4 = 100.)
2. **Challenge:** Give one example of an RL problem where the state space is technically finite but too large for tabular methods (e.g. chess, or a discretized robot state with 10 dimensions and 100 bins per dimension). Estimate the number of states and explain why function approximation is necessary.
