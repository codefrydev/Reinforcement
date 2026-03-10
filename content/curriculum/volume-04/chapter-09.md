---
title: "Chapter 39: Deep Deterministic Policy Gradient (DDPG)"
description: "DDPG for Pendulum with OU noise and target networks."
date: 2026-03-10T00:00:00Z
weight: 39
draft: false
tags: ["DDPG", "Pendulum", "OU noise", "target networks", "curriculum"]
keywords: ["DDPG", "Deep Deterministic Policy Gradient", "Pendulum", "OU noise"]
---

**Learning objectives**

- Implement **DDPG**: **deterministic** policy (actor) plus **Q-function** (critic), with **target networks** and a **replay buffer**.
- Use **Ornstein-Uhlenbeck (OU) noise** (or Gaussian noise) on the action for **exploration** in continuous spaces.
- Train on **Pendulum-v1** (or similar) and plot the learning curve.

**Concept and real-world RL**

**DDPG** is an actor-critic method for **continuous actions**: the actor outputs a single action \\(\mu(s)\\) (no distribution), and the critic learns \\(Q(s,a)\\). The policy is updated to maximize \\(Q(s, \mu(s))\\) (gradient through Q into the actor). Target networks and replay buffer stabilize learning (like DQN). Exploration comes from **adding noise** to the action (e.g. OU noise for temporally correlated exploration, or simple Gaussian). In **robot control** (Pendulum, MuJoCo), DDPG is a baseline for continuous tasks; **TD3** and **SAC** improve on it with clipped double Q and stochastic policies.

**Where you see this in practice:** DDPG is used in robotics, simulated control, and trading. TD3 and SAC are often preferred for better stability and sample efficiency.

**Exercise:** Implement DDPG for the Pendulum-v1 environment. Use actor and critic networks with target networks and a replay buffer. Add Ornstein-Uhlenbeck noise for exploration. Plot the learning curve.

**Professor's hints**

- Replay buffer: store (s, a, r, s', done). Sample mini-batches for training. Buffer size 1e5–1e6.
- Critic: input (s, a), output scalar Q(s,a). Loss = MSE between Q(s,a) and target \\(y = r + \gamma (1-\mathrm{done}) Q_{target}(s', \\mu_{target}(s'))\\). Use target networks for \\(Q_{target}\\) and \\(\mu_{target}\\).
- Actor: input s, output action \\(\mu(s)\\). Loss = -\\(Q(s, \\mu(s))\\) (maximize Q). Gradient flows from Q through \\(\mu(s)\\) into the actor.
- OU noise: \\(d x = \\theta (\\mu - x) dt + \\sigma dW\\); in discrete form, \\(x_{t+1} = x_t + \\theta (\\mu - x_t) + \\sigma N(0,1)\\). Add \\(x\\) to the action. Or use Gaussian noise for simplicity.
- Update targets: soft update \\(\\theta' \\leftarrow \\tau \\theta' + (1-\\tau)\\theta\\) every step, or hard update every N steps.

**Common pitfalls**

- **Forgetting to clamp actions:** Pendulum may expect actions in \\([-2, 2]\\); after adding noise, clip to the env’s action space.
- **No gradient through target:** When computing \\(y\\), use \\(\mu_{target}(s')\\) and \\(Q_{target}(s', a')\\) with `.detach()` so the target is not updated by the current batch.

{{< collapse summary="Worked solution (warm-up: why DDPG needs replay and target)" >}}
**Warm-up:** Replay buffer: to decorrelate consecutive samples (i.i.d. batches) and reuse experience for sample efficiency, same as DQN. Target networks: to stabilize the TD target \\(y = r + \\gamma Q_{target}(s', \\mu_{target}(s'))\\); without a target, the label would change every update and learning would be unstable. DDPG uses soft updates (\\(\\tau\\)) to slowly track the online network.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does DDPG need a replay buffer and target networks? (Same reasons as DQN: decorrelate samples and stabilize targets.)
2. **Coding:** Implement DDPG for Pendulum. Plot episode return every 10 episodes. How many episodes until you reach an average return of -200 or better?
3. **Challenge:** Replace OU noise with **Gaussian noise** (zero mean, constant std). Compare learning speed and final performance. Then try **decaying** the noise std over time (exploration schedule).
