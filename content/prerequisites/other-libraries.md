---
title: "Other Libraries"
description: "JAX, stable-baselines3, wandb, and other RL-related tools."
date: 2026-03-10T00:00:00Z
weight: 80
draft: false
tags: ["JAX", "stable-baselines3", "wandb", "libraries", "prerequisites"]
keywords: ["JAX", "stable-baselines3", "wandb", "RL tools", "other libraries"]
---

Optional tools you may encounter or use alongside the curriculum: JAX for fast autograd and JIT, Stable-Baselines3 for ready-made algorithms, and Weights & Biases for experiment tracking. No need to master these before starting; refer back when an exercise or chapter mentions them.

---

## JAX

- **What:** Autograd and JIT compilation; functional style; used in research (Brax, RLax, many papers).
- **Concepts:** `jax.grad`, `jax.jit`, `jax.vmap`, arrays similar to NumPy. GPU/TPU without explicit device code.
- **When:** Chapters or papers that use JAX-based envs or algorithms.

**Docs:** [jax.readthedocs.io](https://jax.readthedocs.io/).

---

## Stable-Baselines3 (SB3)

- **What:** Implementations of PPO, SAC, DQN, A2C, etc. with a common interface: `model.learn()`, `model.predict()`.
- **Concepts:** `PPO("MlpPolicy", env, ...)`, `model.learn(total_timesteps=1e5)`, `model.predict(obs)`. Callbacks for logging.
- **When:** Baselines in experiments (Chapters 47–48, 51), or when you want to compare your implementation to a reference.

**Docs:** [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/).

---

## Weights & Biases (wandb)

- **What:** Experiment tracking: log metrics, hyperparameters, and plots to the cloud; compare runs in a dashboard.
- **Concepts:** `wandb.init(project="rl", config=config)`, `wandb.log({"reward": r, "step": t})`, sweeps for grid search.
- **When:** Chapter 50 (hyperparameter tuning), and any multi-run experiments.

**Docs:** [wandb.ai](https://wandb.ai/).

---

## Exercises (conceptual and light practice)

**Exercise 1.** (Reading) Skim the JAX "Quickstart" and the section on `jax.grad`. Write one sentence on how `jax.grad` differs from PyTorch’s `backward()` in terms of usage (e.g. function vs. method, return value).

**Exercise 2.** (Reading) Install Stable-Baselines3 and run the "Basic Usage" example for PPO on CartPole (train for a few thousand steps). Note the commands or code you used (e.g. `PPO("MlpPolicy", env, verbose=1).learn(5000)`). Report the final mean episode reward from the console output if available.

**Exercise 3.** (Conceptual) List three things you might log in wandb during an RL training run (e.g. episode return, loss, policy entropy). For each, say whether you’d log it every step, every episode, or every N episodes, and why.

**Exercise 4.** (Reading) Look up the Stable-Baselines3 documentation for "Callback". Name one built-in callback and what it does (e.g. CheckpointCallback). Write a one-line description of how you could use it in an experiment.

**Exercise 5.** (Optional coding) If you have a wandb account, add to a small script: `import wandb`, `wandb.init(project="test", config={"lr": 1e-3})`, and inside a loop `wandb.log({"dummy": i})` for i in range(10). Run the script and confirm a run appears in your project dashboard.

**Exercise 6.** (Reading) In Stable-Baselines3, how do you evaluate a trained model on an environment without exploration? (Hint: `predict` with `deterministic=True`.) Write one sentence. **In RL:** Evaluation should use the greedy policy, not random actions.

**Exercise 7.** (Conceptual) Why might you use JAX instead of PyTorch for a new RL research project? List one advantage (e.g. JIT, functional style, or hardware). When might you prefer PyTorch? **In RL:** Many papers use JAX for speed and portability; PyTorch has more tutorials and ecosystem.

---

## Professor's hints

- **In RL:** Use SB3 when you need a **baseline** quickly (e.g. "How good is PPO on this env?") or when comparing your implementation to a reference. Do not rely on it for every chapter—the curriculum expects you to implement algorithms yourself.
- **wandb:** Log at least episode return and (if applicable) loss every episode or every N steps. Log hyperparameters in `config` so you can compare runs. Do not log every single step for long runs (too much data).
- **JAX:** If a chapter or paper uses JAX, focus on the algorithm idea first; you can reimplement in PyTorch for practice. Learning JAX is optional unless you join a JAX-based lab or project.

---

## Common pitfalls

- **SB3 version and env compatibility:** Different SB3 versions support different Gym/Gymnasium versions. Check the SB3 docs for "Gymnasium" or "Gym" and install the right env package.
- **wandb offline:** If you have no internet, use `wandb.init(..., mode="offline")`; sync later with `wandb sync`.
- **JAX and NumPy:** Do not mix `import numpy as np` and `import jax.numpy as jnp` in the same file without care; use `jnp` for arrays that JAX will differentiate or JIT.

---

No need to learn these in depth before the curriculum; use them when an exercise or chapter refers to them.
