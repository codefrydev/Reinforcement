---
title: "Learning Path: Zero to Reinforcement Learning"
description: "A step-by-step roadmap from no programming experience to building and understanding RL. Start here if you are an absolute beginner."
date: 2026-03-10T00:00:00Z
draft: false
---

This learning path takes you from **zero programming experience** to understanding and building reinforcement learning systems. Follow the phases in order; each phase builds on the previous one.

- **[Real-world scenarios](/learning-path/real-world-anchors/)** — Six anchor scenarios (robot navigation, game AI, recommendation, trading, healthcare, dialogue) used throughout the curriculum so every concept is tied to practice.

**Not ready for the [Preliminary assessment](/preliminary/)?** If you have never programmed, start with Phase 0. If the assessment feels hard, follow this learning path in order and return to it when you are ready.

---

## Phase 0 — Programming from zero

**For:** Anyone with no prior coding experience.

**Duration:** About 2–4 weeks (at a few hours per week).

**What you will do:** Install Python, run your first script, and learn variables, types, conditionals, loops, and functions.

**Outcomes:**

- You can run a Python script and write a small program.
- You understand variables, conditionals, loops, and functions well enough to read simple code.
- You are ready for the full [Python prerequisite](/prerequisites/python/).

**In RL, this leads to:** Every RL implementation is code. You will write loops over episodes, conditionals for exploration vs. exploitation, and functions for environments and agents.

**Start here:** [Phase 0: Programming from zero](/learning-path/phase-0/)

---

## Phase 1 — Math foundations for RL

**For:** Learners who can write basic Python (or have finished Phase 0 and the Python prerequisite) and want to solidify the math used in RL.

**Duration:** About 2–4 weeks.

**What you will do:** Study probability & statistics, linear algebra, and calculus with RL-motivated examples and practice. Work through the sub-phases in order, then take the self-check.

**Sub-phases:**

- **1a — Probability:** [Probability & statistics](/math-for-rl/probability/). Expectations, variance, sample mean, law of large numbers. *In RL:* bandit rewards, Monte Carlo returns, value functions as expectations.
- **1b — Linear algebra:** [Linear algebra](/math-for-rl/linear-algebra/). Vectors, dot product, matrices, gradients. *In RL:* state vectors, linear value approximation \\(V(s) = w^T \\phi(s)\\), gradient updates.
- **1c — Calculus:** [Calculus](/math-for-rl/calculus/). Derivatives, chain rule, partial derivatives. *In RL:* policy gradients, loss minimization, backprop.

**Outcomes:**

- You can read RL notation (expectations, distributions, vectors, gradients).
- You can compute sample means, variances, dot products, and simple derivatives.
- You feel comfortable with the math that appears in the [Preliminary assessment](/preliminary/) and in Volume 1.

**In RL, this leads to:** Value functions are expectations; states and observations are vectors; policy gradients use calculus. Solid math makes every chapter easier.

**Start here:** [Math for RL](/math-for-rl/) → then [Phase 1 self-check](/assessment/phase-1-math/)

---

## Phase 2 — Prerequisites (tools and libraries)

**For:** Learners who have basic programming (and ideally some math) and are ready to use the stack the curriculum assumes.

**Duration:** About 3–6 weeks, depending on how much you already know.

**What you will do:** Work through Python (full), NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, and Gym as needed. Each prerequisite page explains **why RL needs it**; complete the **one small task** per topic listed on the [Prerequisites](/prerequisites/) index, then take the **Phase 2 readiness quiz**.

**Outcomes:**

- You can use the data structures, classes, and patterns used in RL code (trajectories, configs, buffers).
- You can create arrays, do batch operations, and plot results with NumPy and Matplotlib.
- You can define and train small neural networks with PyTorch (or TensorFlow) and run Gym environments.

**In RL, this leads to:** The curriculum exercises assume this stack. Prerequisites include Professor's hints and common pitfalls to avoid mistakes.

**Start here:** [Prerequisites](/prerequisites/) → [Phase 2 readiness quiz](/assessment/phase-2-readiness/)

---

## Phase 3 — RL foundations

**For:** Learners who have completed (or tested out of) Phases 0–2 and are ready for the core RL curriculum.

**Duration:** About 4–8 weeks.

**What you will do:** Complete [Volume 1: Mathematical Foundations](/curriculum/volume-01/) and [Volume 2: Tabular Methods & Classic Algorithms](/curriculum/volume-02/) (chapters 1–20). Use the **milestone checkpoints** and **mini-project** (tabular Q-learning on a 5×5 Gridworld) on the [Phase 3](/learning-path/phase-3/) page, then take the **Phase 3 foundations quiz**.

**Outcomes:**

- You understand the RL framework (agent, environment, state, action, reward), MDPs, and the Markov property.
- You can explain value functions, Bellman equations, and discounting.
- You understand and can implement Monte Carlo, TD, SARSA, and Q-learning in tabular settings.

**In RL, this leads to:** Everything that follows (DQN, policy gradients, etc.) builds on these ideas. Do not skip this phase.

**Start here:** [Volume 1: Mathematical Foundations](/curriculum/volume-01/) → [Phase 3 milestones & mini-project](/learning-path/phase-3/) → [Phase 3 foundations quiz](/assessment/phase-3-foundations/)

---

## Phase 4 — Deep RL

**For:** Learners who have finished Volumes 1–2 and want to scale to large or continuous state spaces.

**Duration:** About 6–12 weeks.

**What you will do:** Complete [Volume 3: Value Function Approximation & Deep Q-Learning](/curriculum/volume-03/), [Volume 4: Policy Gradients](/curriculum/volume-04/), and [Volume 5: Advanced Policy Optimization](/curriculum/volume-05/) (chapters 21–50).

**Outcomes:**

- You can implement and tune DQN-style methods (replay, target networks, etc.) and policy gradient methods (REINFORCE, actor-critic, PPO).
- You understand why function approximation is needed and how gradient-based updates work in RL.

**In RL, this leads to:** Most practical applications use deep RL. This phase is where you go from “understanding the theory” to “building agents that work in complex environments.”

**Start here:** [Volume 3: Value Function Approximation & Deep Q-Learning](/curriculum/volume-03/) → [Phase 4 milestones & coding challenges](/learning-path/phase-4/) → [Phase 4 Deep RL quiz](/assessment/phase-4-deep-rl/)

---

## Phase 5 — Advanced topics

**For:** Learners who have completed Phases 3–4 and want to go deeper.

**Duration:** Ongoing (pick topics as needed).

**What you will do:** Work through [Volumes 6–10](/curriculum/) (chapters 51–100): model-based RL, exploration, offline RL, multi-agent RL, real-world applications, safety, and RL with large language models. Each volume has a **topic roadmap** (what you will learn per chapter); use it to pick a path. An **optional Phase 5 project** (e.g. offline RL on a fixed dataset, or a simple multi-agent scenario) ties concepts together.

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

**Start here:** [Volume 6: Model-Based RL & Planning](/curriculum/volume-06/)

---

## Quick reference

| Phase | Content | Duration (approx.) |
|-------|---------|---------------------|
| 0 | [Programming from zero](/learning-path/phase-0/) | 2–4 weeks |
| 1 | [Math for RL](/math-for-rl/) | 2–4 weeks |
| 2 | [Prerequisites](/prerequisites/) | 3–6 weeks |
| 3 | [Volume 1](/curriculum/volume-01/) + [Volume 2](/curriculum/volume-02/) | 4–8 weeks |
| 4 | [Volumes 3–5](/curriculum/) | 6–12 weeks |
| 5 | [Volumes 6–10](/curriculum/) | Ongoing |

Good luck on your journey from zero to mastery.
