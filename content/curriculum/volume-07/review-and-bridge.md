---
title: "Volume 7 Review & Bridge to Volume 8"
description: "Review Volume 7 (Exploration, ICM, RND, Go-Explore, Meta-RL) and preview Volume 8 (Offline RL, Imitation Learning, RLHF)."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 8
weight: 100
tags: ["review", "bridge", "Volume 7", "Volume 8", "exploration", "ICM", "offline RL", "RLHF"]
roadmap_color: "indigo"
roadmap_icon: "sparkles"
roadmap_phase_label: "Vol 7 · Review"
---

## Volume 7 Recap Quiz (5 questions)

{{< collapse summary="Q1. How does ICM (Intrinsic Curiosity Module) generate an intrinsic reward?" >}}
ICM has two networks: (1) a **forward model** that predicts the next feature state given (s_t, a_t); (2) an **inverse model** that predicts action a_t from (φ(s_t), φ(s_{t+1})). The intrinsic reward is the **prediction error of the forward model**: r_i = ||φ̂(s_{t+1}) − φ(s_{t+1})||². The agent is "curious" about states where its forward model is surprised. The inverse model learns features that are sensitive to agent-controllable aspects of the environment (ignoring uncontrollable noise like TV static).
{{< /collapse >}}

{{< collapse summary="Q2. What problem does RND (Random Network Distillation) solve that ICM does not?" >}}
ICM can get "distracted" by stochastic or uncontrollable parts of the environment (the "noisy TV" problem — a random TV always produces high prediction error). RND avoids this: it trains a small network to predict the output of a **fixed random target network** on each state. Novel states produce high prediction error; visited states produce low error. Because the target is deterministic, stochastic noise doesn't create spuriously high novelty.
{{< /collapse >}}

{{< collapse summary="Q3. What is the two-phase approach of Go-Explore?" >}}
Phase 1 (**Explore**): maintain an archive of interesting states; repeatedly return to a promising archived state (ignoring any learned policy — direct reset) and explore randomly from there. This detaches exploration from the current policy. Phase 2 (**Robustify**): once a path to the goal is found, train a policy to reliably follow it using imitation learning. This separation allows finding solutions to hard games that confound end-to-end RL.
{{< /collapse >}}

{{< collapse summary="Q4. What does count-based exploration do, and why doesn't it scale?" >}}
Count-based methods maintain N(s) — the visit count for each state — and add bonus r_+ = β / √N(s). Well-visited states get small bonuses; novel states get large bonuses. This works well in small tabular MDPs but **fails in large or continuous state spaces** where almost every state is unique (N(s)=1 always). Pseudo-counts and density models extend this idea to high-dimensional spaces.
{{< /collapse >}}

{{< collapse summary="Q5. What is MAML (Model-Agnostic Meta-Learning) and what is its goal in RL?" >}}
MAML finds an initial parameter θ such that a **few gradient steps** on a new task yields good performance. It explicitly optimises for fast adaptability: θ* = argmax_θ E_τ[L(θ − α ∇L(θ, D_τ))]. In RL, each "task" is a different environment configuration. At test time, the agent can adapt to a new task with just 1–3 episodes. This is different from standard RL which trains from scratch per environment.
{{< /collapse >}}

---

## What Changes in Volume 8

| | Volume 7 (Online RL — Interactive) | Volume 8 (Offline RL — Fixed Dataset) |
|---|---|---|
| **Data collection** | Agent interacts with environment | Dataset is fixed — no new interaction |
| **Exploration** | Core challenge | Not possible (fixed data) |
| **Distribution shift** | Manageable (agent controls policy) | Critical — OOD actions can be catastrophic |
| **Key risk** | Getting stuck in local optima | Overestimating Q for unvisited (s,a) |
| **Key methods** | ICM, RND, Go-Explore | BCQ, CQL, Decision Transformer, IRL |

**The big insight:** Sometimes you have a large logged dataset (from humans or previous policies) but cannot run new experiments — medical devices, autonomous vehicles, expensive robots. Offline RL learns from this fixed data without any environment interaction. The challenge: the Q-function may wildly overestimate value for actions never seen in the dataset ("out-of-distribution" actions).

---

## Bridge Exercise: The Offline RL Distribution Shift Problem

{{< pyrepl code="# Why is offline RL hard? Illustrate Q overestimation on OOD actions.\nimport random\nrandom.seed(42)\n\n# Imagine a 1D state, 3 possible actions: 0, 1, 2\n# Dataset only contains actions 0 and 1 (action 2 never tried)\n\ntrue_Q = {0: 1.0, 1: 2.0, 2: 5.0}   # true Q-values (action 2 is actually TERRIBLE)\ndata_actions = [0, 1, 0, 1, 1, 0]    # actions in the offline dataset\n\n# Naive Q-learning from offline data can't evaluate action 2\n# But if we use function approximation, it may EXTRAPOLATE:\n\nQ_estimated = {0: 0.9, 1: 1.8, 2: 8.0}  # network extrapolates action 2 upward!\n\nprint('=== Offline Dataset Actions ===')\nprint(f'Actions seen in data: {sorted(set(data_actions))}')\nprint(f'Action 2 never seen  -- no data to correct its Q estimate')\nprint()\nprint(f'{'Action':<8} {'True Q':>8} {'Estimated Q':>12} {'Error':>8}')\nfor a in [0, 1, 2]:\n    err = Q_estimated[a] - true_Q[a]\n    flag = ' <-- OOD! Q is wildly overestimated' if a == 2 else ''\n    print(f'{a:<8} {true_Q[a]:>8.1f} {Q_estimated[a]:>12.1f} {err:>+8.1f}{flag}')\n\nprint()\nprint('Greedy policy on estimated Q would choose action 2 -- catastrophic!')\nprint('CQL / BCQ add penalties for OOD actions to prevent this.')" height="320" >}}

**Next:** [Volume 8: Offline RL & Imitation Learning](../volume-08/)
