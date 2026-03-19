---
title: "Chapter 35: Actor-Critic Architectures"
description: "Sketch two-network actor-critic; pseudocode for TD error updates."
date: 2026-03-10T00:00:00Z
weight: 35
draft: false
tags: ["actor-critic", "TD error", "advantage", "curriculum"]
keywords: ["actor-critic", "two-network", "TD error", "advantage"]
---

**Learning objectives**

- Sketch the **architecture** of a two-network actor-critic: **actor** (policy \\(\pi(a|s)\\)) and **critic** (value \\(V(s)\\) or \\(Q(s,a)\\)).
- Write **pseudocode** for the update steps using the **TD error** \\(\delta = r + \gamma V(s') - V(s)\\) as the advantage for the policy.
- Explain why the critic reduces variance compared to using Monte Carlo returns \\(G_t\\).

**Concept and real-world RL**

**Actor-critic** methods maintain two networks: the **actor** selects actions from \\(\pi(a|s;\theta)\\), and the **critic** estimates the value function \\(V(s;w)\\) (or Q). The **TD error** \\(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\\) is a one-step estimate of the advantage; it is biased (because V is approximate) but much lower variance than \\(G_t\\). The actor is updated with \\(\nabla \log \pi(a_t|s_t) \, \delta_t\\); the critic is updated to minimize \\((r_t + \gamma V(s_{t+1}) - V(s_t))^2\\). In **robot control** and **game AI**, actor-critic allows online, step-by-step updates instead of waiting for episode end, which speeds up learning.

**Where you see this in practice:** A2C, A3C, and many continuous control algorithms (DDPG, TD3, SAC) are actor-critic style. The pattern (policy + value/advantage) is central to PPO as well.

**Illustration (actor and critic outputs):** The critic estimates \\(V(s)\\); the actor outputs action probabilities. The chart below shows example \\(V(s)\\) for a few states and the policy entropy over training.

{{< chart type="line" palette="return" title="V(s) and entropy over episodes" labels="Ep 0, Ep 50, Ep 100, Ep 150" data="0, 15, 35, 50" xLabel="Episode" yLabel="Value" >}}

**Exercise:** Sketch the architecture of a two-network actor-critic: the actor outputs a distribution over actions, the critic outputs a value. Write pseudocode for the update steps using TD error.

**Professor's hints**

- Sketch: state \\(s\\) → shared or separate layers → actor head (logits → softmax → \\(\pi(a|s)\\)) and critic head (scalar \\(V(s)\\)). Or two separate networks: one for \\(\pi\\), one for V.
- Pseudocode: (1) sample \\(a \sim \pi(\cdot|s)\\), step env → \\(s', r, \mathrm{done}\\); (2) \\(\delta = r + \gamma (1-\mathrm{done}) V(s') - V(s)\\); (3) actor loss = \\(-\log \pi(a|s) \, \delta\\); critic loss = \\(\delta^2\\); (4) backward on both, update. Use \\(V(s').detach()\\) in \\(\delta\\) for the actor so gradients do not go through the target.
- TD error replaces \\(G_t\\): one number per step instead of a sum over the whole episode, so variance is typically lower.

**Common pitfalls**

- **Gradient through target:** When computing \\(\delta = r + \gamma V(s')\\), do not backprop through \\(V(s')\\) when updating the actor (use `.detach()`), otherwise the actor would try to change the critic’s target.
- **Critic and actor learning rates:** If the critic learns too fast, V can be noisy and \\(\delta\\) becomes unreliable. Often the critic uses a separate optimizer or a smaller learning rate.

{{< collapse summary="Worked solution (warm-up: TD error and bias)" >}}
**Warm-up:** \\(\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)\\). It is a biased estimate of \\(A(s_t,a_t)\\) because \\(V(s_{t+1})\\) is an approximation; the true advantage uses the true value function. So \\(\\delta_t\\) has lower variance than \\(G_t\\) (one step of randomness) but is biased. Actor-critic trades off: we use \\(\\delta_t\\) (or n-step) as the advantage estimate to reduce variance while accepting some bias.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Write the TD error \\(\delta_t\\) in terms of \\(r_t, \gamma, V(s_t), V(s_{t+1})\\). Why is \\(\delta_t\\) a biased estimate of the advantage \\(A(s_t,a_t)\\)?
2. **Coding:** Implement a minimal actor-critic: one-step update (single transition). Given \\(s, a, r, s'\\) and networks \\(\pi, V\\), compute \\(\delta\\), actor loss, critic loss, and one gradient step. Test with dummy tensors.
3. **Challenge:** Extend your pseudocode to **n-step TD**: use \\(\delta_t = G_{t:t+n} - V(s_t)\\) where \\(G_{t:t+n} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})\\). How does n affect bias and variance?
4. **Variant:** Try \\(\gamma=0.9\\) and \\(\gamma=0.99\\) in the TD error. Does the critic converge to the correct \\(V\\) faster with lower \\(\gamma\\)? Why?
5. **Debug:** The code below computes the actor loss but allows gradient to flow through \\(V(s')\\), coupling the actor and critic updates incorrectly. Fix it with `.detach()`.

{{< pyrepl code="import torch\n\ndef actor_critic_update(actor, critic, s, a, r, s_next, gamma=0.99):\n    V_s = critic(s)\n    V_s_next = critic(s_next)  # BUG: no .detach()\n    # delta flows gradient through V_s_next into critic via actor loss\n    delta = r + gamma * V_s_next - V_s  # V_s_next should be detached\n    log_prob = actor(s).log_prob(a)\n    actor_loss = -log_prob * delta  # BUG: delta has critic gradient\n    critic_loss = delta ** 2\n    return actor_loss + critic_loss\n\n# Fix: V_s_next = critic(s_next).detach()\nprint('Fix: detach V(s_next) before computing delta for actor loss')" height="220" >}}

6. **Conceptual:** Why does the TD error \\(\\delta_t\\) reduce variance compared to the Monte Carlo return \\(G_t\\)? What bias does it introduce, and where does that bias come from?
7. **Recall:** Write actor-critic pseudocode for one step (sample action, observe transition, compute \\(\\delta\\), update actor, update critic) from memory.
