---
title: "Chapter 78: Adversarial Motion Priors (AMP)"
description: "AMP paper: task reward + adversarial style reward; combined reward."
date: 2026-03-10T00:00:00Z
weight: 78
draft: false
tags: ["AMP", "adversarial motion priors", "imitation", "style reward", "curriculum"]
keywords: ["AMP", "Adversarial Motion Priors", "task reward", "adversarial style reward"]
---

**Learning objectives**

- **Read** the AMP paper and **explain** how it combines a **task reward** (e.g. velocity tracking, goal reaching) with an **adversarial style reward** (discriminator that scores motion similarity to reference data).
- **Write** the **combined reward** function: r = r_task + λ r_style, where r_style comes from a discriminator trained to distinguish agent motion from reference (e.g. motion capture) data.
- **Identify** why adding a style reward helps produce natural-looking and robust locomotion compared to task-only reward.
- **Relate** AMP to **robot navigation** and **game AI** (character animation) where we want both task success and natural motion.

**Concept and real-world RL**

**Adversarial Motion Priors (AMP)** train policies that achieve a **task** (e.g. run forward, follow a path) while matching **reference motion** (e.g. from motion capture). A **discriminator** is trained to tell "reference" motion (short sequences of state or pose) from "policy" motion; the policy gets a **style reward** for fooling the discriminator. The **combined reward** is r = r_task + λ r_style. This encourages the agent to solve the task with motions that look like the reference, avoiding unnatural behaviors that might get high task reward. In **robot navigation** and **game AI**, this is used for humanoid locomotion and character control so that the agent moves naturally while achieving goals.

**Where you see this in practice:** AMP and related methods for locomotion; motion capture–driven RL; style + task reward in animation.

**Illustration (AMP style reward):** AMP combines task reward with an adversarial style reward for natural motion. The chart below shows task return and style score over training.

{{< chart type="line" title="Task return and style score (AMP)" labels="0, 500, 1000, 1500, 2000" data="0, 200, 500, 800, 1200" >}}

**Exercise:** Read the AMP paper and explain how it combines a task reward with an adversarial style reward to produce natural locomotion. Write the combined reward function.

**Professor's hints**

- **Task reward:** e.g. r_task = -|v - v*|^2 for velocity tracking, or +1 for reaching a goal. This is the usual RL objective.
- **Style reward:** Discriminator D takes a short window of state/pose (e.g. 1–2 seconds of joint positions and velocities). Reference data = motion capture or hand-designed motion. Agent rollouts provide "fake" windows. r_style = -log(1 - D(window)) or log D(window) so the policy is rewarded for producing windows that look like reference. Combined: r = r_task + λ * r_style; λ tunes the trade-off.
- **Paper details:** AMP typically uses a discriminator on **motion snippets** (not single timesteps), and the policy is trained with RL (e.g. PPO) on the combined reward. The discriminator is updated to distinguish reference vs policy snippets.
- For the "write the combined reward" part, provide the formula and a short explanation of each term; implementation of full AMP is optional but recommended for practice.

**Common pitfalls**

- **λ too high:** If the style reward dominates, the agent may ignore the task (e.g. stand still in a natural pose). Tune λ so both task and style matter.
- **Discriminator on single frames:** Single timesteps may not capture "motion"; AMP uses **temporal windows** so the discriminator sees dynamics, not just pose.
- **Reference distribution:** The reference data should be diverse enough (e.g. many motion capture clips) so the policy does not overfit to one motion.

{{< collapse summary="Worked solution (warm-up: imitation from reference)" >}}
**Key idea:** We have reference trajectories (e.g. motion capture). We train a policy to match the reference state distribution or state-action distribution (e.g. supervised loss, or reward = similarity to reference). The policy learns to reproduce the reference behavior. For diverse references we get a multi-modal policy; for a single reference we get cloning. Used in character animation and robot manipulation.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might a policy that only maximizes task reward (e.g. forward speed) produce unnatural or unstable motion? How does adding a style reward help?
2. **Coding:** Implement the combined reward: r = r_task + λ r_style. Use a simple 2D or 3D locomotion env (e.g. HalfCheetah). For r_style, use a discriminator on (s_t, s_{t+1}) or a 3-step window. Get "reference" by running a random policy and taking a subset of transitions (or use a scripted gait). Train PPO with r and vary λ ∈ {0, 0.1, 1.0}. Compare motion quality (e.g. visually or via jerk/smoothness).
3. **Challenge:** Reproduce a minimal version of AMP on a MuJoCo humanoid: task = move forward; reference = a few motion capture clips (or scripted walk). Train with PPO + style discriminator and report task return and style score (discriminator output on policy data).
