---
title: "Chapter 40: Twin Delayed DDPG (TD3)"
description: "TD3: clipped double Q, delayed policy, target smoothing."
date: 2026-03-10T00:00:00Z
weight: 40
draft: false
---

**Learning objectives**

- Implement **TD3** improvements over DDPG: **two critics** (clipped double Q-learning), **delayed policy updates** (update actor less often than critic), and **target policy smoothing** (add noise to the target action).
- Compare **performance** on a continuous control task (e.g. HalfCheetah if feasible, or Pendulum / BipedalWalker) with vanilla DDPG.

**Concept and real-world RL**

**TD3** (Twin Delayed DDPG) addresses DDPG’s overestimation and instability: (1) **Two Q-networks**: take the minimum of the two Q-values for the target (like Double DQN), reducing overestimation. (2) **Delayed policy updates**: update the actor every \\(d\\) critic updates so the critic is more accurate before the actor is trained. (3) **Target policy smoothing**: add small Gaussian noise to \\(\mu_{target}(s')\\) when computing the target, so the target is less sensitive to the exact action. In **robot control** and **simulated benchmarks** (HalfCheetah, Hopper), TD3 often achieves better and more stable performance than DDPG.

**Where you see this in practice:** TD3 is a standard baseline in continuous control (e.g. OpenAI Spinning Up, CleanRL). SAC extends the idea further with entropy regularization.

**Exercise:** Enhance your DDPG with TD3 improvements: two critics (clipped double Q-learning), delayed policy updates, and target policy smoothing. Compare performance on a continuous control task like HalfCheetah (if feasible) or a simpler environment.

**Professor's hints**

- Two critics: maintain \\(Q_1, Q_2\\) and two targets \\(Q_1', Q_2'\\). Target value: \\(y = r + \gamma (1-\mathrm{done}) \, \min(Q_1'(s', a'), Q_2'(s', a'))\\) where \\(a' = \mu_{target}(s') + \epsilon\\), \\(\epsilon \sim \mathrm{clip}(\\mathcal{N}(0, \\tilde{\\sigma}), -c, c)\\) (target smoothing).
- Critic loss: train both \\(Q_1\\) and \\(Q_2\\) on the same target \\(y\\) (MSE). So both critics try to predict the same conservative target.
- Delayed policy: update actor every \\(d\\) steps (e.g. \\(d=2\\)). Each step, update critics; every \\(d\\)-th step, also update the actor to maximize \\(Q_1(s, \\mu(s))\\) (or average of \\(Q_1, Q_2\\)).
- HalfCheetah: if heavy to run, use Pendulum or BipedalWalker for comparison. Plot mean return over 10 eval episodes every 1000 steps.

**Common pitfalls**

- **Using max instead of min for target:** TD3 uses \\(\min(Q_1', Q_2')\\) to reduce overestimation. Using max would revert to DDPG-style overestimation.
- **Target smoothing too large:** If \\(\tilde{\\sigma}\\) is too big, the target becomes very noisy. Typical values are small (e.g. 0.2 * action scale).

**Extra practice**

1. **Warm-up:** In one sentence each, what is the purpose of (a) clipped double Q, (b) delayed policy updates, (c) target policy smoothing?
2. **Coding:** Implement TD3 for Pendulum. Compare with your DDPG: plot both learning curves on the same figure. Does TD3 reach a higher return or converge faster?
3. **Challenge:** Run TD3 on **HalfCheetah-v4** (or v3). Tune learning rate and policy delay \\(d\\). Report the average return over the last 10 episodes after 1M steps (or your compute limit).
