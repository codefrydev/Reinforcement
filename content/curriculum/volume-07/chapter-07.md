---
title: "Chapter 67: Meta-Learning (Learning to Learn)"
description: "Task distribution (e.g. goal positions); meta-training loop, few-step adapt."
date: 2026-03-10T00:00:00Z
weight: 67
draft: false
---

**Learning objectives**

- **Define** a distribution of tasks (e.g. different goal positions in a gridworld) and sample tasks for meta-training.
- **Implement** a meta-training loop: for each task, collect data or run a few steps of adaptation, then update the meta-policy or meta-parameters to improve few-task performance.
- **Explain** the goal of meta-RL: learn an initialization or algorithm that adapts quickly to new tasks with few gradient steps or few episodes.
- **Evaluate** the meta-learned policy on held-out tasks with limited data and compare with training from scratch.
- **Relate** meta-RL to **robot navigation** (different goals or terrains) and **game AI** (different levels or opponents).

**Concept and real-world RL**

**Meta-learning** in RL aims to **learn to learn**: the agent is trained on a **distribution of tasks** (e.g. different goal positions, different dynamics, or different reward functions) so that it can **adapt quickly** to a new task from the same distribution with few episodes or few gradient steps. In **robot navigation**, tasks might be "reach goal A" vs "reach goal B" or different maps; in **game AI**, tasks might be different levels or game modes. The meta-training loop typically samples a task, runs the current policy for a few steps (or one inner update), and then updates the meta-parameters to minimize loss or maximize return across tasks. This chapter focuses on the loop and task distribution rather than a specific meta-algorithm like MAML.

**Where you see this in practice:** MAML, RL², and similar meta-RL methods; few-shot adaptation in robotics and games.

**Exercise:** Define a distribution of tasks (e.g., different goal positions in a gridworld). Write the meta-training loop for a model that can adapt quickly to a new task with a few gradient steps.

**Professor's hints**

- **Task distribution:** e.g. sample goal (row, col) uniformly in a gridworld; each task is "reach this goal." Or sample different reward weights or wall layouts.
- **Meta-loop:** Outer loop: sample a batch of tasks. For each task: (1) run the current policy for K steps or one inner update, (2) compute loss or return on that task. Aggregate over tasks and take an outer gradient step to update the policy (or its initialization).
- "Few gradient steps" can mean: inner loop does a few steps of policy gradient or supervised update on the task; outer loop updates the initial parameters so that after the inner steps, performance is good. Start with a simple inner loop (e.g. one policy gradient step per task).
- Use a small gridworld (e.g. 5×5) and 2–5 inner steps so you can debug and see adaptation.

**Common pitfalls**

- **Inner loop too long:** If each task gets many inner steps, the meta-learner may overfit to "easy" tasks; keep inner steps small to emphasize fast adaptation.
- **Task distribution mismatch:** If test tasks are very different from training tasks, meta-learning may not transfer; keep the same distribution for train and test (e.g. same grid, different goals).
- **Second-order gradients:** Some meta-RL methods need gradients through the inner update; for a first implementation, a first-order approximation (treat inner update as fixed) is simpler and often sufficient.

**Extra practice**

1. **Warm-up:** Why is it useful to train on many tasks (e.g. many goal positions) instead of one fixed task if we want an agent that can quickly learn new goals?
2. **Coding:** Implement a task distribution (e.g. 10 goal positions in a 5×5 grid). Train a shared policy with a meta-loop: sample task, run 3 steps, compute return, update policy. Evaluate on 5 held-out goals with 5 steps of adaptation. Plot return vs number of outer iterations.
3. **Challenge:** Use **MAML-style** inner update: one gradient step on the task loss with respect to the policy parameters. Compute the meta-gradient (gradient of post-adaptation loss w.r.t. initial parameters). Compare with first-order (no gradient through inner step).
