---
title: "Other Libraries"
description: "JAX, stable-baselines3, wandb, and other RL-related tools."
date: 2026-03-10T00:00:00Z
weight: 80
draft: false
difficulty: 6
tags: ["JAX", "stable-baselines3", "wandb", "libraries", "prerequisites"]
keywords: ["JAX", "stable-baselines3", "wandb", "RL tools", "other libraries"]
roadmap_icon: "globe"
roadmap_color: "blue"
roadmap_phase_label: "Phase 6 · Other Tools"
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

## Hands-on coding exercises

**Exercise 8 (JAX — discounted return).** Implement a discounted return function in NumPy (runs in browser). The JAX equivalent with `@jax.jit` is shown in the comments — run it locally after installing JAX.

{{< pyrepl code="# JAX is not available in browser — run locally:\n# pip install jax\n# import jax\n# import jax.numpy as jnp\n# \n# @jax.jit\n# def discounted_return(rewards, gamma=0.9):\n#     return sum(gamma**t * r for t, r in enumerate(rewards))\n# \n# But we can compare: here is the NumPy equivalent (runs in browser)\nimport numpy as np\n\ndef discounted_return_np(rewards, gamma=0.9):\n    # TODO: implement using numpy\n    pass\n\nprint(discounted_return_np([0, 0, 1], 0.9))  # expected: 0.81" height="240" >}}

*Note: Run the JAX version locally. The NumPy version above runs in browser.*

**Exercise 9 (Stable-Baselines3 — training loop pseudocode).** Write the pseudocode for what `model.learn(10000)` does under the hood. Fill in the TODO steps.

{{< pyrepl code="# Pseudocode for what SB3 model.learn(10000) does:\n# Run this to understand the training loop conceptually\n\ndef sb3_learn_pseudocode(total_timesteps=10000):\n    timestep = 0\n    episode = 0\n    while timestep < total_timesteps:\n        # TODO: fill in the steps:\n        # 1. env.reset() at start of episode\n        # 2. while not done:\n        #    a. agent.predict(obs)\n        #    b. env.step(action)\n        #    c. store transition in buffer\n        #    d. if buffer full: policy.update()\n        # 3. increment counters\n        print(f'Episode {episode}: {timestep}/{total_timesteps} timesteps')\n        timestep += 100  # simplified\n        episode += 1\n        if episode >= 3: break  # just for demo\n\nsb3_learn_pseudocode()" height="280" >}}

**Exercise 10 (Weights & Biases — minimal logger).** Implement a simple experiment logger that mimics `wandb.log()` behavior. Fill in the `log` and `summary` methods.

{{< pyrepl code="# A minimal experiment logger (mimics wandb.log behavior)\n# Run locally with wandb: import wandb; wandb.init(project='rl-exp'); wandb.log({'reward': r})\n\nclass SimpleLogger:\n    def __init__(self, run_name):\n        self.run_name = run_name\n        self.history = []\n    \n    def log(self, metrics):\n        # TODO: append metrics dict to self.history\n        pass\n    \n    def summary(self):\n        # TODO: print last 3 entries from self.history\n        pass\n\n# Test it\nlogger = SimpleLogger('my-experiment')\nfor ep in range(5):\n    logger.log({'episode': ep, 'reward': ep * 10, 'loss': 1.0 / (ep + 1)})\nlogger.summary()" height="280" >}}

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
