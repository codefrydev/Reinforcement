---
title: "Learning Path: Zero to Reinforcement Learning"
description: "A step-by-step roadmap from no programming experience to building and understanding RL. Start here if you are an absolute beginner."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["learning path", "roadmap", "beginner", "zero to RL", "reinforcement learning"]
keywords: ["learning path", "zero to reinforcement learning", "beginner roadmap", "RL from scratch", "step-by-step RL"]
roadmap_icon: "rocket"
roadmap_color: "indigo"
roadmap_phase_label: "Learning Path"
---

This learning path takes you from **zero programming experience** to understanding and building reinforcement learning systems. Follow the phases in order; each phase builds on the previous one. The order matches the [Course outline](../course-outline/) (basic to advanced).

- **[Real-world scenarios](real-world-anchors/)** — Six anchor scenarios (robot navigation, game AI, recommendation, trading, healthcare, dialogue) used throughout the curriculum so every concept is tied to practice.

**Not ready for the [Preliminary assessment](../preliminary/)?** If you have never programmed, start with Phase 0. If the assessment feels hard, follow this learning path in order and return to it when you are ready.

---

## Phase 0 — Programming from zero

**For:** Anyone with no prior coding experience.

**Duration:** About 2–4 weeks (at a few hours per week).

**What you will do:** Install Python, run your first script, and learn variables, types, conditionals, loops, and functions.

**Outcomes:**

- You can run a Python script and write a small program.
- You understand variables, conditionals, loops, and functions well enough to read simple code.
- You are ready for the full [Python prerequisite](../prerequisites/python/).

**In RL, this leads to:** Every RL implementation is code. You will write loops over episodes, conditionals for exploration vs. exploitation, and functions for environments and agents.

**Start here:** [Phase 0: Programming from zero](phase-0/)

**New in this curriculum:**
- [Python Confidence Builder](phase-0/python-confidence/) — 25 mini-challenges before Phase 1 (Level 0.5)
- [RL in Plain English](rl-in-plain-english/) — no math, no code intuition builder (Level 1.5)
- [Bridge Exercises](bridge-exercises/) — 15 problems combining Python + math + toy RL (Level 2.5)

---

## Phase 1 — Math foundations for RL

**For:** Learners who can write basic Python (or have finished Phase 0 and the Python prerequisite) and want to solidify the math used in RL.

**Duration:** About 2–4 weeks.

**What you will do:** Study probability & statistics, linear algebra, and calculus with RL-motivated examples and practice. Work through the sub-phases in order, then take the self-check.

**Sub-phases:**

- **1a — Probability:** [Probability & Statistics](../math-for-rl/probability/). Expectations, variance, sample mean, law of large numbers. *In RL:* bandit rewards, Monte Carlo returns, value functions as expectations.
- **1b — Statistics:** [Statistics for RL](../math-for-rl/statistics/). Mean, variance, standard deviation, standard error, histograms, correlation. *In RL:* analyzing episode returns, reporting results with error bars.
- **1c — Linear algebra:** [Linear algebra](../math-for-rl/linear-algebra/). Vectors, dot product, matrices, gradients. *In RL:* state vectors, linear value approximation \\(V(s) = w^T \\phi(s)\\), gradient updates.
- **1d — Calculus:** [Calculus](../math-for-rl/calculus/). Derivatives, chain rule, partial derivatives. *In RL:* policy gradients, loss minimization, backprop.

**Outcomes:**

- You can read RL notation (expectations, distributions, vectors, gradients).
- You can compute sample means, variances, dot products, and simple derivatives.
- You feel comfortable with the math that appears in the [Preliminary assessment](../preliminary/) and in Volume 1.

**In RL, this leads to:** Value functions are expectations; states and observations are vectors; policy gradients use calculus. Solid math makes every chapter easier.

**Start here:** [Math for RL](../math-for-rl/) → then [Phase 1 self-check](../assessment/phase-1-math/)

---

## Phase 2 — Prerequisites (tools and libraries)

**For:** Learners who have basic programming (and ideally some math) and are ready to use the stack the curriculum assumes.

**Duration:** About 3–6 weeks, depending on how much you already know.

**What you will do:** Work through Python (full), NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, and Gym as needed. Each prerequisite page explains **why RL needs it**; complete the **one small task** per topic listed on the [Prerequisites](../prerequisites/) index, then take the **Phase 2 readiness quiz**.

**Outcomes:**

- You can use the data structures, classes, and patterns used in RL code (trajectories, configs, buffers).
- You can create arrays, do batch operations, and plot results with NumPy and Matplotlib.
- You can define and train small neural networks with PyTorch (or TensorFlow) and run Gym environments.

**In RL, this leads to:** The curriculum exercises assume this stack. Prerequisites include Professor's hints and common pitfalls to avoid mistakes.

**Start here:** [Prerequisites](../prerequisites/) → [Phase 2 readiness quiz](../assessment/phase-2-readiness/)

---

## Phase 3 — Math for RL (deep dive)

**For:** Learners who have completed Phases 0–2 and want a deeper foundation before RL.

**Duration:** About 2–3 weeks.

**What you will do:** Complete [Math for RL](../math-for-rl/) — probability, statistics, linear algebra, and calculus with RL-motivated examples. Work through the drills at the end of each math page.

**Outcomes:**

- You can interpret RL notation fluently.
- You understand why value functions are expectations, why gradients are used for optimization, and how to compute sample statistics from RL evaluation runs.

**Start here:** [Math for RL](../math-for-rl/) index

---

## Phase 4 — ML Foundations

**For:** Learners who want to understand supervised learning before tackling deep RL.

**Duration:** About 3–5 weeks.

**What you will do:** Complete [ML Foundations](../ml-foundations/) — supervised learning, linear/logistic regression, gradient descent, model evaluation, decision trees, and KNN. Take the checkpoint at the midpoint and the Phase 4 assessment at the end.

**Outcomes:**

- You understand how supervised learning trains models from labeled data using gradient descent.
- You can implement and evaluate linear regression, logistic regression, and simple classifiers.
- You know how to split data, evaluate with metrics (accuracy, precision, recall, F1), and avoid data leakage.

**In RL, this leads to:** Deep RL builds on these foundations. DQN uses supervised-style regression on Q-values; policy networks are trained with gradient descent. Understanding supervised learning makes deep RL much clearer.

**Start here:** [ML Foundations](../ml-foundations/) → [ML Mid-Point Checkpoint](../assessment/checkpoint-ml-mid/) → [Phase 4 assessment](../assessment/phase-4-ml/)

---

## Phase 5 — DL Foundations

**For:** Learners who have completed ML Foundations and are ready for neural networks.

**Duration:** About 4–6 weeks.

**What you will do:** Complete [DL Foundations](../dl-foundations/) — biological inspiration, perceptrons, MLP, backpropagation, loss functions, activations, optimizers, training loops, regularization, CNNs, PyTorch, and the mini-project. Take the DL mid-point checkpoint and Phase 5 assessment.

**Outcomes:**

- You can implement a full neural network from scratch in NumPy (forward pass, loss, backprop, gradient update).
- You understand how Adam, SGD, and Momentum work and when to use each.
- You can build a QNetwork and PolicyNetwork in PyTorch using `nn.Module`.
- You are ready to implement DQN and policy gradient methods.

**In RL, this leads to:** Everything in deep RL. DQN, REINFORCE, PPO, and actor-critic all use neural networks with the exact patterns you built here.

**Start here:** [DL Foundations](../dl-foundations/) → [DL Mid-Point Checkpoint](../assessment/checkpoint-dl-mid/) → [Phase 5 assessment](../assessment/phase-5-dl/)

---

## Phase 6 — RL foundations

**For:** Learners who have completed (or tested out of) Phases 0–5 and are ready for the core RL curriculum.

**Duration:** About 4–8 weeks.

**What you will do:** Complete [Volume 1: Mathematical Foundations](../curriculum/volume-01/) and [Volume 2: Tabular Methods & Classic Algorithms](../curriculum/volume-02/) (chapters 1–20). Use the **milestone checkpoints** and **mini-project** (tabular Q-learning on a 5×5 Gridworld) on the [Phase 6](phase-3/) page, then take the **Phase 6 foundations quiz**.

**Outcomes:**

- You understand the RL framework (agent, environment, state, action, reward), MDPs, and the Markov property.
- You can explain value functions, Bellman equations, and discounting.
- You understand and can implement Monte Carlo, TD, SARSA, and Q-learning in tabular settings.

**In RL, this leads to:** Everything that follows (DQN, policy gradients, etc.) builds on these ideas. Do not skip this phase.

**Start here:** [Volume 1: Mathematical Foundations](../curriculum/volume-01/) → [Phase 6 milestones & mini-project](phase-3/) → [Phase 6 foundations quiz](../assessment/phase-3-foundations/)

---

## Phase 7 — Deep RL

**For:** Learners who have finished Volumes 1–2 and want to scale to large or continuous state spaces.

**Duration:** About 6–12 weeks.

**What you will do:** Complete [Volume 3: Value Function Approximation & Deep Q-Learning](../curriculum/volume-03/), [Volume 4: Policy Gradients](../curriculum/volume-04/), and [Volume 5: Advanced Policy Optimization](../curriculum/volume-05/) (chapters 21–50).

**Outcomes:**

- You can implement and tune DQN-style methods (replay, target networks, etc.) and policy gradient methods (REINFORCE, actor-critic, PPO).
- You understand why function approximation is needed and how gradient-based updates work in RL.

**In RL, this leads to:** Most practical applications use deep RL. This phase is where you go from "understanding the theory" to "building agents that work in complex environments."

**Start here:** [Volume 3: Value Function Approximation & Deep Q-Learning](../curriculum/volume-03/) → [Phase 7 milestones & coding challenges](phase-4/) → [Phase 7 Deep RL quiz](../assessment/phase-4-deep-rl/)

---

## Phase 8 — Advanced topics

**For:** Learners who have completed Phases 6–7 and want to go deeper.

**Duration:** Ongoing (pick topics as needed).

**What you will do:** Work through [Volumes 6–10](../curriculum/) (chapters 51–100): model-based RL, exploration, offline RL, multi-agent RL, real-world applications, safety, and RL with large language models. Each volume has a **topic roadmap** (what you will learn per chapter); use it to pick a path. An **optional Phase 8 project** (e.g. offline RL on a fixed dataset, or a simple multi-agent scenario) ties concepts together.

**Topic roadmaps (after this you will…):**

- **Vol 6 (Model-based):** Compare model-free vs model-based; learn world models and compounding error; implement planning (BFS, MCTS), Dreamer-style imagination, MBPO, PETS.
- **Vol 7 (Exploration & meta):** Tackle hard exploration (sparse rewards); intrinsic motivation, curiosity (ICM), RND; Go-Explore; meta-learning (MAML, RL²).
- **Vol 8 (Offline & imitation):** Understand offline RL (distribution shift, CQL); Decision Transformers; behavioral cloning, DAgger, IRL, GAIL; RLHF basics.
- **Vol 9 (Multi-agent):** Game theory basics; IQL, CTDE, MADDPG; VDN, QMIX; MAPPO; self-play; communication.
- **Vol 10 (Real-world, safety, LLMs):** Robotics and sim-to-real; safe RL; trading, recommenders; PPO/RLHF for LLMs; evaluation and debugging.

**Optional project:** Implement offline RL on a fixed dataset (e.g. from a random or expert policy) using CQL or conservative Q-learning; or implement a simple two-agent cooperative task with parameter sharing (MAPPO or IQL).

**Outcomes:**

- You can read RL papers and extend the project.
- You understand model-based methods, exploration, offline and imitation learning, MARL, and how RL is used in practice (robotics, trading, recommenders, RLHF).

**In RL, this leads to:** Research and industry applications. Use the curriculum as a map and dive into the areas that interest you most.

**Start here:** [Volume 6: Model-Based RL & Planning](../curriculum/volume-06/)

---

## Quick reference

| Phase | Content | Duration (approx.) |
|-------|---------|---------------------|
| 0 | [Programming from zero](phase-0/) | 2–4 weeks |
| 1 | [Math for RL](../math-for-rl/) | 2–4 weeks |
| 2 | [Prerequisites](../prerequisites/) | 3–6 weeks |
| 3 | [Math for RL (deep dive)](../math-for-rl/) | 2–3 weeks |
| 4 | [ML Foundations](../ml-foundations/) | 3–5 weeks |
| 5 | [DL Foundations](../dl-foundations/) | 4–6 weeks |
| 6 | [Volume 1](../curriculum/volume-01/) + [Volume 2](../curriculum/volume-02/) | 4–8 weeks |
| 7 | [Volumes 3–5](../curriculum/) | 6–12 weeks |
| 8 | [Volumes 6–10](../curriculum/) | Ongoing |

Good luck on your journey from zero to mastery.

---

## Quick Reference

- [Glossary](../glossary/) — 75 RL terms with definitions, chapter references, and examples
- [Assessments](../assessment/) — Phase 0 through Phase 8, mid-point checkpoints
- [Appendix: Debugging RL Code](../appendix/debugging-rl-code/) — Common bugs and 5 find-the-bug exercises
- [Appendix: Reading RL Papers](../appendix/reading-rl-papers/) — How to read DQN, PPO, and SAC papers
- [Interactive Lab]({{< laburl >}}) — Run Python in your browser (JupyterLite)
