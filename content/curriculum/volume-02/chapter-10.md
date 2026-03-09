---
title: "Chapter 20: The Limits of Tabular Methods"
description: "Memory for Backgammon Q-table; necessity of function approximation."
date: 2026-03-10T00:00:00Z
weight: 20
draft: false
---

**Learning objectives**

- Estimate memory for a tabular Q-table (states × actions × bytes per entry).
- Relate the scale of real problems (e.g. Backgammon, continuous state) to the infeasibility of tables.
- Argue why function approximation (linear, neural) is necessary for large or continuous spaces.

**Concept and real-world RL**

**Tabular methods** store one value per state (or state-action). When the state space is huge or continuous, this is impossible: Backgammon has on the order of \\(10^{20}\\) states; a robot with 10 continuous state variables discretized to 100 bins each has \\(100^{10}\\) cells. **Function approximation** uses a parameterized function \\(Q(s,a; \\theta)\\) with a fixed number of parameters, so we can represent values for infinitely many states. This is the bridge to deep RL (Volumes 3–5): neural networks are function approximators that generalize from visited states to unseen ones.

**Exercise:** Estimate the memory required to store a Q-table for the game of Backgammon (approx. \\(10^{20}\\) states) with 4-byte floats. Compare this with the memory of a modern computer. Discuss the necessity of function approximation.

**Professor's hints**

- Backgammon: assume \\(\\approx 10^{20}\\) states and a small number of actions (e.g. on the order of 10–100). Q-table size = states × actions × 4 bytes. So roughly \\(10^{20} \\times 50 \\times 4\\) bytes ≈ \\(2 \\times 10^{22}\\) bytes. Convert to TB: \\(2 \\times 10^{22} / 10^{12}\\) ≈ \\(10^{10}\\) TB. A typical computer has 8–64 GB RAM (\\(10^9\\)–\\(10^{10}\\) bytes); so the table is astronomically larger.
- Compare: 1 TB = \\(10^{12}\\) bytes. Your estimate in TB shows that no physical machine can store the table. Hence we need a compact representation (function approximation).
- Discussion: function approximation uses a fixed number of parameters (e.g. millions for a neural net) to represent \\(Q\\) for all states. We learn \\(\\theta\\) from experience so that \\(Q(s,a;\\theta)\\) generalizes to unseen \\(s\\). Trade-off: we can handle huge spaces but may have approximation error and instability.

**Common pitfalls**

- **Underestimating state count:** Backgammon has many board configurations and dice outcomes; \\(10^{20}\\) is a common cited order of magnitude. Do not use a tiny number (e.g. 1000) for "Backgammon."
- **Forgetting actions:** Q-table is states × actions. Backgammon has a variable number of legal moves; use an upper bound (e.g. a few hundred) for a rough estimate.
- **Claiming FA has no downsides:** Function approximation can diverge, generalize poorly, or forget; it is necessary for scale but introduces new challenges (covered in Volumes 3–5).

**Extra practice**

1. **Warm-up:** A 10×10 grid has 100 states and 4 actions. How many floats for a Q-table with 4 bytes each? (100 × 4 × 4 = 1600 bytes.)
2. **Challenge:** Estimate the number of parameters in a small neural network that takes a Backgammon state (encoded as a vector of size 100) and outputs Q-values for 50 actions (one hidden layer of 256 units). Compare that number to the tabular size. (E.g. 100×256 + 256×50 ≈ 38k parameters; 38k × 4 bytes ≈ 150 KB.)
