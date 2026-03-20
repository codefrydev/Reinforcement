---
title: "Volume 6 Review & Bridge to Volume 7"
description: "Review Volume 6 (Model-Based RL, MCTS, Dyna-Q, world models) and preview Volume 7 (Exploration — intrinsic motivation, curiosity, and sparse rewards)."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 8
weight: 100
tags: ["review", "bridge", "Volume 6", "Volume 7", "model-based", "MCTS", "exploration", "curiosity"]
roadmap_color: "rose"
roadmap_icon: "brain"
roadmap_phase_label: "Vol 6 · Review"
---

## Volume 6 Recap Quiz (5 questions)

{{< collapse summary="Q1. What are the four phases of MCTS, and what happens in each?" >}}
1. **Selection**: traverse the tree from root using UCB1 (balance exploit/explore) until a leaf.
2. **Expansion**: add one or more child nodes to the leaf.
3. **Simulation** (rollout): play out randomly (or with a fast policy) until a terminal state.
4. **Backpropagation**: update win counts / values along the path from leaf to root.

AlphaZero replaces random rollouts with a learned value network, eliminating phase 3.
{{< /collapse >}}

{{< collapse summary="Q2. How does Dyna-Q combine model-free and model-based learning?" >}}
After each real interaction (s,a,r,s'), Dyna-Q: (1) updates Q directly (model-free TD), (2) updates the learned model M(s,a) → (r,s'), then (3) performs k planning steps: sample (s,a) from memory, query M, and update Q again with synthetic transitions. This reuses real experience multiple times, improving sample efficiency.
{{< /collapse >}}

{{< collapse summary="Q3. What is the 'compounding error' problem in model-based RL?" >}}
A learned model M(s,a) has some prediction error ε per step. Over an n-step rollout, errors compound: the agent may end up in regions of state space the model has never seen (distribution shift), producing wildly incorrect predictions. This limits how far ahead you can reliably plan with a learned model. Short rollouts (MBPO: 1–5 steps) mitigate this at the cost of reduced planning horizon.
{{< /collapse >}}

{{< collapse summary="Q4. What is a world model, and how does Dreamer use it?" >}}
A world model learns a compressed latent representation of environment dynamics: a recurrent state model s_{t+1} ~ p(s_{t+1}|s_t, a_t), a reward model r ~ p(r|s_t), and a decoder. Dreamer trains the policy **entirely inside the model's imagination** — generating multi-step latent rollouts without any real environment interaction. Real data is only used to update the world model itself.
{{< /collapse >}}

{{< collapse summary="Q5. What is the key assumption that model-based methods make that exploration challenges?" >}}
Model-based methods assume the learned model is **accurate across the state space the agent will visit**. But if the agent only visits states near the start, the model is only accurate there. Exploration is needed to visit diverse states so the model generalises. Without good exploration, the agent may optimise against a model that is wildly wrong in unvisited regions — exploitation of model errors.
{{< /collapse >}}

---

## What Changes in Volume 7

| | Volume 6 (Model-Based, Dense Rewards) | Volume 7 (Hard Exploration, Sparse Rewards) |
|---|---|---|
| **Reward signal** | Assumed frequent / dense | Sparse or deceptive — rare signal |
| **Exploration** | ε-greedy or entropy bonus suffice | Dedicated exploration: ICM, RND, Go-Explore |
| **State coverage** | Incidental | Actively maximised (count-based, novelty) |
| **Challenge** | Model accuracy / planning horizon | Finding reward at all |
| **Examples** | MuJoCo locomotion | Montezuma's Revenge, maze navigation |

**The big insight:** When rewards are sparse, the agent may never stumble upon a positive signal by random exploration. Intrinsic motivation — curiosity, novelty, prediction error — provides a dense internal reward signal that drives exploration independent of the task reward.

---

## Bridge Exercise: How Quickly Does Random Exploration Fail?

{{< pyrepl code="import random\n\nrandom.seed(42)\n\ndef random_maze_search(grid_size, max_steps):\n    \"\"\"\n    Agent starts at (0,0), goal at (grid_size-1, grid_size-1).\n    Random walk -- how many steps to reach goal?\n    \"\"\"\n    goal = (grid_size - 1, grid_size - 1)\n    pos = (0, 0)\n    for step in range(max_steps):\n        dr = random.choice([-1, 0, 1])\n        dc = random.choice([-1, 0, 1])\n        nr = max(0, min(grid_size-1, pos[0] + dr))\n        nc = max(0, min(grid_size-1, pos[1] + dc))\n        pos = (nr, nc)\n        if pos == goal:\n            return step + 1\n    return None  # never found\n\n# Test random search on increasingly large grids\nfor size in [5, 10, 20, 50]:\n    trials = [random_maze_search(size, 100_000) for _ in range(20)]\n    found = [t for t in trials if t is not None]\n    pct = 100 * len(found) / len(trials)\n    avg = sum(found)/len(found) if found else float('inf')\n    print(f'Grid {size:2d}x{size:2d}: found in {pct:3.0f}% of trials, avg steps={avg:,.0f}')\n\nprint()\nprint('Random exploration scales poorly -- needs intrinsic motivation at scale!')" height="300" >}}

**Next:** [Volume 7: Exploration & Intrinsic Motivation](../volume-07/)
