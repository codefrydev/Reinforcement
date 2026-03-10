---
title: "Chapter 99: Debugging RL Code"
description: "Broken SAC: unit tests, logging Q/reward/entropy; diagnose."
date: 2026-03-10T00:00:00Z
weight: 99
draft: false
tags: ["debugging", "RL", "curriculum"]
keywords: ["debugging RL", "RL code", "troubleshooting"]
---

**Learning objectives**

- **Take** a **broken** RL implementation (e.g. SAC that does not learn or converges to poor return) and **diagnose** the issue systematically.
- **Write** **unit tests** for the environment (e.g. step returns correct shapes, reset works, reward is bounded), the replay buffer (e.g. sample returns correct batch shape, storage and sampling are consistent), and **gradient shapes** (e.g. critic loss backward produces gradients of the right shape).
- **Add** **logging** for Q-values (min, max, mean), rewards (per step and per episode), and **entropy** (or log_prob) so you can spot numerical issues, collapse, or scale problems.
- **Identify** the root cause (e.g. wrong sign, wrong target, learning rate, or reward scale) and fix it.
- **Relate** debugging practice to **robot navigation** and **healthcare** where bugs can be costly.

**Concept and real-world RL**

**Debugging RL code** is hard because the learning signal is noisy and many components interact (env, buffer, critic, actor, target network). A **systematic** approach: (1) **Unit test** each piece: env step/reset, buffer add/sample, and that gradients flow (loss.backward() and optimizer.step() change parameters). (2) **Log** key quantities: Q(s,a) range, reward distribution, entropy. (3) **Sanity checks**: reward scale (not too large/small), Q not exploding, policy not deterministic too early. (4) **Compare** with a known-good implementation or a simpler env. In **robot navigation** and **healthcare**, catching bugs before deployment is critical.

**Where you see this in practice:** Unit testing in RL libraries; logging and monitoring in training loops; debugging checklists in RL courses and blogs.

**Exercise:** Take a broken RL implementation (e.g., SAC that doesn't learn). Write unit tests for the environment, replay buffer, and gradient shapes. Add logging for Q-values, rewards, and entropy. Diagnose the issue.

**Professor's hints**

- **Broken SAC:** You can introduce a bug (e.g. wrong sign in actor loss, target Q not detached, reward multiplied by 0, or learning rate too high/low) or use an existing buggy snippet. The goal is to find and fix it.
- **Unit tests:** (1) Env: reset() returns obs in expected shape; step(a) returns (obs, reward, done, info); run 10 steps and check no NaN. (2) Buffer: add 100 transitions, sample batch of 32; check shapes of s, a, r, s', done. (3) Gradients: compute critic loss, loss.backward(); check that critic parameters have non-zero grad; same for actor.
- **Logging:** Every N steps, log mean Q(s,a) on a batch, mean reward in the last 10 episodes, mean entropy (or mean log π(a|s)). Plot or print; look for Q going to 0 or infinity, reward always 0, entropy collapsing to 0.
- **Common SAC bugs:** Forgetting to detach target Q; wrong sign in the actor loss (should maximize Q); reward scale (e.g. need to normalize); replay buffer not filling before updates.

**Common pitfalls**

- **Assuming the env is correct:** Often the bug is in the env (e.g. wrong reward, done flag, or state). Test the env in isolation.
- **Not checking gradients:** If a loss term is not connected to the graph (e.g. constant), gradients will be zero and that part of the model will not learn.
- **Overlooking scale:** Rewards or Q-values that are too large can cause instability; too small and learning is slow. Log and normalize if needed.

{{< collapse summary="Worked solution (warm-up: LLM scale)" >}}
**Key idea:** When applying RL to large language models, we have huge state/action spaces (vocabulary, long sequences). We need to scale rewards and gradients: e.g. normalize advantages, clip rewards, use a small KL penalty coefficient so the policy doesn’t collapse. Training is expensive so we reuse a pretrained model and do a few PPO (or DPO) updates. Monitor reward and KL; if reward spikes and KL explodes, we are reward hacking—tune the KL penalty.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why is it useful to log Q-values and entropy during SAC training? What would you suspect if Q-values grow without bound?
2. **Coding:** Start from a minimal SAC (or use a known buggy version). Add unit tests for env and buffer. Add logging for Q, reward, entropy. Run for 10k steps. If it does not learn, use the logs and tests to find the bug; fix and verify that return increases.
3. **Challenge:** Intentionally introduce **three** bugs (e.g. wrong target, missing detach, reward scale). Write a short "debugging report": what you logged, what you observed, and how you identified each bug.
