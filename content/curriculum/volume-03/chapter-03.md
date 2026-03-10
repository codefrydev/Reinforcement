---
title: "Chapter 23: Deep Q-Networks (DQN)"
description: "DQN for CartPole with replay and target network."
date: 2026-03-10T00:00:00Z
weight: 23
draft: false
tags: ["DQN", "CartPole", "experience replay", "target network", "curriculum"]
keywords: ["Deep Q-Networks", "DQN", "replay buffer", "target network", "CartPole"]
---

**Learning objectives**

- Implement full DQN: Q-network, target network, replay buffer, \\(\epsilon\\)-greedy, and the TD loss (MSE to target \\(r + \\gamma \\max_{a'} Q_{target}(s',a')\\)).
- Update the target network periodically (e.g. every 100 steps) by copying the online Q-network.
- Train on CartPole and plot reward per episode.

**Concept and real-world RL**

**DQN** combines a neural network for Q-values with **experience replay** (store transitions, sample random minibatches to break correlation) and a **target network** (separate copy of the network used in the TD target, updated periodically, to stabilize learning). The agent acts \\(\epsilon\\)-greedy, stores \\((s,a,r,s',\\text{done})\\) in the buffer, and repeatedly samples a batch, computes targets using the target network, and updates the online network by minimizing MSE. DQN was the first major deep RL success (Atari) and is still a standard baseline for discrete-action tasks.

**Exercise:** Implement DQN for the CartPole-v1 environment. Use a replay buffer of size 10,000, target network update every 100 steps, and \\(\epsilon\\)-greedy exploration. Train for 500 episodes and plot the rewards.

**Professor's hints**

- Replay buffer: store (s, a, r, s', done). When buffer has at least batch_size (e.g. 64) samples, sample a batch and do one gradient step. Use a circular buffer (e.g. list with max length, or a NumPy array and index).
- Target: \\(y = r + \\gamma (1 - \\text{done}) \\max_{a'} Q_{target}(s', a')\\). When done=1, target = r. Use the target network for \\(Q_{target}(s', a')\\); do not backprop through it (detach).
- Update target: every 100 env steps (or 100 gradient steps), copy online params to target: `target.load_state_dict(online.state_dict())`. Decay \\(\\epsilon\\) from 1.0 to 0.05 or 0.1 over training if you want.

**Common pitfalls**

- **Backprop through target:** The target \\(y\\) must be detached. If you do `loss = mse_loss(Q(s,a), r + gamma * Q_target(s',a').max())`, the target part should not have gradients (use `.detach()` on the target tensor).
- **Done flag:** When done is True, the target is just \\(r\\) (no next state). So \\(y = r + \\gamma (1 - \\text{done}) \\max_{a'} Q_{target}(s',a')\\). For done=1 this gives \\(y = r\\).
- **Replay before learning:** Do not perform gradient updates until the buffer has enough samples (e.g. at least batch_size). Early on, just collect experience.

{{< collapse summary="Worked solution (warm-up: DQN target y)" >}}
**Warm-up:** For one transition (s, a, r, s', done=0), write the target \\(y\\) in terms of \\(Q_{target}\\) and \\(\\gamma\\). For done=1, what is \\(y\\)? **Answer:** For done=0: \\(y = r + \\gamma \\max_{a'} Q_{target}(s', a')\\). For done=1 (terminal): \\(y = r\\) (no bootstrap). So in code: `y = r + (1 - done) * gamma * Q_target(s').max(dim=1)[0]`. We use the target network so the label is stable during training.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For one transition (s, a, r, s', done=0), write the target \\(y\\) in terms of \\(Q_{target}\\) and \\(\\gamma\\). For done=1, what is \\(y\\)?
2. **Coding:** Implement the DQN loss (MSE between Q(s,a) and target y) for a batch. Use a target network for y; compute y with no_grad. Test with dummy tensors.
3. **Challenge:** Add **double DQN**: use the online network to select the action \\(a^* = \\arg\\max_a Q(s',a)\\), but use \\(Q_{target}(s', a^*)\\) as the target value. Compare learning curve with standard DQN on CartPole.
