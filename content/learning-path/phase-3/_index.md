---
title: "Phase 3: RL Foundations — Milestones & Mini-Project"
description: "Checkpoints after Vol 1 and Vol 2; mini-project: tabular Q-learning on Gridworld."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["phase 3", "foundations", "Q-learning", "Gridworld", "milestones", "learning path"]
keywords: ["RL foundations", "Volume 1 Volume 2", "tabular Q-learning", "Gridworld", "milestones", "mini-project"]
---

Phase 3 is [Volume 1: Mathematical Foundations](../../curriculum/volume-01/) and [Volume 2: Tabular Methods](../../curriculum/volume-02/) (chapters 1–20). Use the milestones below to confirm you are on track, then do the mini-project and take the foundations quiz.

---

## Milestone checkpoints

- **After Volume 1, Chapter 5 (Bellman equations):** You can write the Bellman expectation equation for \\(V^\\pi(s)\\) by hand and solve for \\(V\\) in a 2-state MDP. You understand that value is expected discounted return.
- **After Volume 1, Chapter 10 (Limitations of DP):** You can explain why tabular methods do not scale to large or continuous state spaces and how function approximation generalizes.
- **After Volume 2, Chapter 5 (SARSA, Q-learning):** You can implement MC prediction (first-visit) and understand the difference between TD(0) and Monte Carlo (bootstrapping vs full return).
- **After Volume 2, Chapter 10 (Scaling to large spaces):** You can implement **Q-learning** in a small gridworld (e.g. 5×5) and extract the greedy policy and value function. You are ready for the mini-project below.

---

## Mini-project: Tabular Q-learning on a 5×5 Gridworld

**Goal:** Implement tabular Q-learning on a 5×5 gridworld. Plot the value function and policy; report the number of steps to reach the goal (averaged over 100 evaluation episodes) after training.

**Specification:**

- **Environment:** 5×5 grid. Start at (0,0), goal at (4,4). Actions: up, down, left, right. Reward: -1 per step, +10 at goal; episode ends at goal or after 50 steps. Hitting a wall leaves the agent in place with -1.
- **Algorithm:** Q-learning with ε-greedy (e.g. ε=0.1), learning rate α=0.1, γ=0.99. Train for 2000 episodes (or until convergence).
- **Deliverables:** (1) A heatmap of \\(V^*(s)\\) (max_a Q(s,a)) at the end of training. (2) A visualization of the greedy policy (arrows or action labels per cell). (3) Mean steps to goal over 100 evaluation episodes (ε=0).

**Hints:**

- Use a dict or 2D array for Q[(s, a)]; state s can be (row, col). Initialize Q to 0.
- After each episode, optionally decay ε (e.g. ε = max(0.01, 0.99 * ε)).
- For the heatmap, use matplotlib (e.g. `plt.imshow(V_grid)`). For the policy, plot arrows in each cell (up/down/left/right).

**Rubric:** You have succeeded if (a) the greedy policy from (0,0) reaches (4,4) without obvious loops, and (b) the value at (0,0) is negative and increases as you get closer to the goal.

When you are done, take the **[Phase 3 foundations quiz](../../assessment/phase-3-foundations/)**.
