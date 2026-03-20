---
title: "Chapter 77: Generative Adversarial Imitation Learning (GAIL)"
description: "Discriminator expert vs agent; use as reward for policy gradient."
date: 2026-03-10T00:00:00Z
weight: 77
draft: false
difficulty: 8
tags: ["GAIL", "imitation", "discriminator", "policy gradient", "curriculum"]
keywords: ["GAIL", "Generative Adversarial Imitation Learning", "discriminator", "expert vs agent"]
roadmap_color: "violet"
roadmap_icon: "database"
roadmap_phase_label: "Vol 8 · Ch 7"
---

**Learning objectives**

- **Implement** GAIL: train a **discriminator** D(s, a) to distinguish state-action pairs from the **expert** vs from the **current policy**; use the discriminator output (or log D) as **reward** for a policy gradient method.
- **Train** the policy to maximize the discriminator reward (i.e. to fool the discriminator) while the discriminator tries to tell expert from agent.
- **Test** on a simple task (e.g. CartPole or MuJoCo) and compare imitation quality with behavioral cloning.
- **Explain** the connection to GANs: the policy is the generator, the discriminator gives the learning signal.
- **Relate** GAIL to **robot navigation** and **game AI** where we have expert demos and want to match the expert distribution without hand-designed rewards.

**Concept and real-world RL**

**Generative Adversarial Imitation Learning (GAIL)** frames imitation as a **generative adversarial** problem: a **discriminator** D(s, a) is trained to classify whether (s, a) came from the expert or from the current policy; the **policy** is trained (via policy gradient) to maximize the reward r(s, a) = log D(s, a) or -log(1 - D(s, a)), i.e. to produce state-action pairs that look like the expert. No explicit reward function is needed—the discriminator provides the learning signal. In **robot navigation** and **game AI**, this allows imitating from demonstrations when the true reward is unknown or hard to specify.

**Where you see this in practice:** GAIL and GAIL-like adversarial imitation; learning from demonstrations without reward engineering.

**Illustration (GAIL discriminator):** The discriminator distinguishes expert from policy data; its accuracy often stays between 0.6–0.8 when both are learning. The chart below shows policy return and discriminator accuracy.

{{< chart type="line" palette="return" title="GAIL: policy return and disc accuracy" labels="0, 100, 200, 300, 400" data="50, 120, 180, 220, 250" xLabel="Iteration" yLabel="Return" >}}

**Exercise:** Implement GAIL: train a discriminator to distinguish between state-action pairs from the expert and from the agent. Use the discriminator output as reward for a policy gradient method. Test on a simple task.

**Professor's hints**

- **Discriminator:** Binary classifier D(s, a) ∈ [0, 1]. Input: (s, a) or concatenation. Label 1 for expert, 0 for agent. Loss: binary cross-entropy. Use a small MLP.
- **Reward for policy:** r(s, a) = log D(s, a) (so the policy wants to maximize D). Or use -log(1 - D(s, a)) for stability. In practice, often use log D - log(1 - D) or just reward = log D so that expert-like (s, a) get high reward.
- **Policy update:** Collect trajectories with the current policy; label them 0. Expert trajectories label 1. Update discriminator on both. Then run policy gradient (e.g. PPO or TRPO) with reward = f(D(s,a)). Repeat.
- **Stability:** Do not update the discriminator to perfection (it would give zero gradient); use a moderate number of D steps per policy step. Optionally use gradient penalty or label smoothing.

**Common pitfalls**

- **Discriminator too strong:** If D quickly distinguishes expert from agent, the reward for the policy becomes near 0 or 1 and the gradient is weak. Use fewer D updates, or reward = log D so the policy always gets a useful gradient when D(s,a) < 1.
- **Mode collapse:** The policy might find a small set of (s, a) that fool D instead of covering the full expert distribution. Use enough expert data and consider adding entropy bonus to the policy.
- **Reward scale:** log D can be very negative when D is small; scale or clip the reward so policy gradient updates are stable.

{{< collapse summary="Worked solution (warm-up: GAIL)" >}}
**Key idea:** GAIL: train a discriminator \\(D(s,a)\\) to distinguish expert (s,a) from policy (s,a). The policy gets reward \\(r = -\\log D(s,a)\\) (or \\(\\log(1-D)\\)) so it tries to fool the discriminator—i.e. match the expert distribution. We do not need to recover an explicit reward; we directly imitate via the discriminator. Use policy gradient (e.g. PPO) with this reward; the discriminator is updated with the binary cross-entropy loss.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why do we use the discriminator output as the reward for the policy? What does "fooling the discriminator" mean in terms of behavior?
2. **Coding:** Implement GAIL on CartPole. Collect 50 expert trajectories from a trained policy. Train a discriminator (MLP, 2 layers) and PPO with reward = log D(s,a). Run for 500 policy iterations. Plot policy return and discriminator accuracy (should stay between 0.6–0.8 if both are learning).
3. **Challenge:** Add a **gradient penalty** to the discriminator (e.g. penalize gradient norm on interpolated expert-agent samples) to improve stability. Compare training curves with and without it.
4. **Variant:** Reduce the expert demonstrations from 50 to 5 trajectories. Does GAIL still converge? Compare with behavioral cloning on the same 5 demonstrations in terms of sample efficiency and final return.
5. **Debug:** GAIL training collapses: the discriminator achieves 100% accuracy after 10 updates and the policy never improves. The discriminator is updated 10 times per policy step, while the policy is updated once. Describe the correct update ratio and explain why an overpowered discriminator starves the policy of learning signal.
6. **Conceptual:** GAIL is equivalent to an occupancy measure matching approach — it minimizes the Jensen-Shannon divergence between the expert's and agent's state-action distributions. Why is matching the occupancy measure a stronger imitation objective than matching the action distribution at observed states (as in BC)?
