---
title: "Chapter 45: Coding PPO from Scratch"
description: "Full PPO for LunarLanderContinuous with GAE and rollout buffer."
date: 2026-03-10T00:00:00Z
weight: 45
draft: false
tags: ["PPO", "LunarLander", "GAE", "rollout buffer", "curriculum"]
keywords: ["PPO implementation", "LunarLanderContinuous", "GAE", "rollout buffer"]
---

**Learning objectives**

- Implement a **full PPO agent** for **LunarLanderContinuous-v2**: policy (actor) and value (critic) networks, rollout buffer, GAE for advantages, and multiple **epochs** of **minibatch** updates per rollout.
- **Tune** key hyperparameters (learning rate, clip \\(\epsilon\\), GAE \\(\lambda\\), batch size, number of epochs) to achieve successful landings.
- Relate each component (clip, GAE, value loss, entropy bonus) to stability and sample efficiency.

**Concept and real-world RL**

**PPO** in practice: collect a rollout of transitions (e.g. 2048 steps), compute GAE advantages, then perform several epochs of minibatch updates on the same data (policy loss with clip + value loss + entropy bonus). The **rollout buffer** stores states, actions, rewards, log-probs, and values; after each rollout we compute advantages and then iterate over minibatches. **LunarLanderContinuous** is a 2D landing task with continuous thrust; it is a standard testbed for PPO. In **robot control** and **game AI**, this "collect rollout → multiple PPO epochs" loop is the core of most on-policy algorithms.

**Where you see this in practice:** LunarLander and similar envs are used in tutorials and benchmarks; the same PPO structure scales to MuJoCo and Atari.

**Illustration (PPO on LunarLander):** Episode return typically improves over training, with some variance. The chart below shows a typical learning curve (mean return per 10 episodes).

{{< chart type="line" palette="return" title="Episode return (PPO LunarLander)" labels="0, 100, 200, 300, 400" data="-200, 50, 150, 220, 250" xLabel="Episode" yLabel="Return" >}}

**Exercise:** Implement a full PPO agent for the LunarLanderContinuous-v2 environment. Use a rollout buffer, compute advantages via GAE, and perform multiple epochs of minibatch updates. Tune hyperparameters to achieve successful landing.

**Professor's hints**

- Rollout: run the policy for N steps (e.g. 2048), store (s, a, r, log_prob, V(s), done). Then compute returns and GAE from rewards and V(s). Append V(s) for the last state (or 0 if done).
- Update: for K epochs (e.g. 4–10), shuffle and split the rollout into minibatches. For each minibatch, compute ratio = π(a|s) / π_old(a|s), clipped loss, value loss (MSE to returns), entropy; total loss = -L_CLIP + c1 * value_loss - c2 * entropy. Backward and step.
- LunarLanderContinuous: state dim 8, action dim 2 (main engine, side boosters). Reward is positive for landing, negative for crashing and fuel. Success: land without crashing and get positive total return.

**Common pitfalls**

- **Reusing old log_probs:** You must store \\(\\log \\pi_{old}(a|s)\\) during rollout and use it for the ratio \\(r_t = \\pi(a|s) / \\pi_{old}(a|s)\\). Do not recompute the old policy after updating.
- **Advantage normalization:** Normalize advantages (zero mean, unit var) per rollout so the scale does not depend on return magnitude; helps with learning rate.

{{< collapse summary="Worked solution (warm-up: PPO objective)" >}}
**Key idea:** PPO maximizes \\(\\mathbb{E}[ \\min(r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t) ]\\) where \\(r_t = \\pi_\\theta(a_t|s_t)/\\pi_{old}(a_t|s_t)\\). We also add an entropy bonus so the policy stays exploratory. We run multiple epochs on the same batch (with clipping) instead of one pass like A2C; that improves sample efficiency while keeping updates safe.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why do we do multiple epochs of updates on the same rollout data? What is the risk if we do too many epochs?
2. **Coding:** Implement PPO for LunarLanderContinuous. Plot episode return every 10 episodes. How many episodes until you first get a successful landing (positive return)?
3. **Challenge:** Ablate: (a) remove the entropy bonus; (b) set \\(\\epsilon = 0\\) (no clip). How does learning stability and final performance change?
4. **Variant:** Change rollout length from 2048 to 512 or 4096 steps. Does a shorter rollout hurt GAE accuracy? Does a longer one slow learning per update? Compare learning curves.
5. **Debug:** The code below stores the log-probs from the *updated* network during the ratio computation — it should use the *old* log-probs stored during rollout. Fix it.

{{< pyrepl code="# Simulated PPO minibatch update\nimport torch\n\ndef ppo_ratio_buggy(actor, states, actions, old_log_probs, eps=0.2):\n    # BUG: recomputes old_log_probs from current network!\n    new_log_probs = actor(states).log_prob(actions)\n    old_log_probs_wrong = actor(states).log_prob(actions)  # same network!\n    ratio = torch.exp(new_log_probs - old_log_probs_wrong)  # ratio ~ 1 always\n    return ratio\n\ndef ppo_ratio_fixed(actor, states, actions, old_log_probs, eps=0.2):\n    # Fix: old_log_probs should be stored during rollout and passed in\n    new_log_probs = actor(states).log_prob(actions)\n    ratio = torch.exp(new_log_probs - old_log_probs)  # correct\n    return ratio\n\nprint('Bug: ratio will always be ~1.0 (no policy update detected)')\nprint('Fix: pass old_log_probs stored at rollout time')" height="240" >}}

6. **Conceptual:** Why does PPO apply multiple gradient steps on the same rollout batch while REINFORCE only does one? Why does the clipping make it safe to do multiple steps?
7. **Recall:** List the four components of the full PPO loss (policy loss, value loss, entropy bonus, and their signs/coefficients) from memory.
