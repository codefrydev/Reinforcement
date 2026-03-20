---
title: "Chapter 10: Limitations of Dynamic Programming"
description: "State and transition count for 10×10 gridworld; function approximation."
date: 2026-03-10T00:00:00Z
weight: 10
draft: false
difficulty: 6
tags: ["dynamic programming", "function approximation", "tabular limits", "curriculum"]
keywords: ["limitations of DP", "function approximation", "state count", "tabular methods"]
roadmap_color: "blue"
roadmap_icon: "layers"
roadmap_phase_label: "Vol 1 · Ch 10"
---

**Learning objectives**

- Compute the number of states and transition probabilities for a small finite MDP.
- Explain why tabular methods (storing a value per state or state-action) do not scale.
- Describe how function approximation (e.g. linear or neural) generalizes across states.

**Concept and real-world RL**

**Dynamic programming** (policy iteration, value iteration) assumes we can store a value for every state (or state-action) and iterate over all of them. In a 10×10 grid that is 100 states—manageable. In real problems (continuous state spaces, or discrete but huge spaces like board games or high-dimensional sensors), the number of states is enormous or infinite, so we cannot store a table.

**Illustration (state count growth):** The number of states grows quickly with grid size. Doubling the side length quadruples the state count. The chart below shows state count for 4×4, 10×10, and 100×100 grids.

{{< chart type="bar" title="Number of states |S| by grid size" labels="4×4, 10×10, 100×100" data="16, 100, 10000" >}}

**Function approximation** uses a parameterized function (e.g. \\(V(s; \\theta)\\) or \\(Q(s,a; \\theta)\\)) so that a fixed number of parameters \\(\\theta\\) represent values for *all* states; we learn \\(\\theta\\) from data. This is the bridge to deep RL (DQN, policy gradients) in Volumes 3–5.

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
3. **Coding:** Write a Python function `qtable_size(rows, cols, num_actions, bytes_per_entry=4)` that returns the number of Q-table entries and the total memory in bytes. Test for a 10×10 grid (4 actions) and a 100×100 grid.

{{< pyrepl code="def qtable_size(rows, cols, num_actions, bytes_per=4):\n    entries = rows * cols * num_actions\n    memory = entries * bytes_per\n    return entries, memory\n\nfor dims in [(10, 10), (100, 100), (1000, 1000)]:\n    r, c = dims\n    entries, mem = qtable_size(r, c, 4)\n    print(f'{r}x{c} grid: {entries} entries, {mem/1e6:.2f} MB')" height="200" >}}

4. **Variant:** A robot has 6 joint angles, each discretized into 50 bins, with 4 possible actions. Compute the total state count and the Q-table memory (in GB). Why is this infeasible?
5. **Debug:** The estimate below forgets to multiply by the number of actions, reporting state count instead of Q-table size. Fix it: `q_size = rows * cols * bytes_per_entry`.
6. **Conceptual:** Name two specific challenges that function approximation introduces compared to tabular methods (e.g. convergence, generalization error).
7. **Recall:** State in one sentence why tabular RL cannot scale to large or continuous state spaces.
