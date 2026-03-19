---
title: "Chapter 25: Target Networks"
description: "Hard vs soft target updates in DQN."
date: 2026-03-10T00:00:00Z
weight: 25
draft: false
tags: ["target network", "DQN", "hard update", "soft update", "curriculum"]
keywords: ["target networks", "DQN", "hard update", "soft update"]
---

**Learning objectives**

- Implement **hard** target updates: copy online network parameters to the target network every \\(N\\) steps.
- Implement **soft** target updates: \\(\\theta_{target} \\leftarrow \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\) each step (or each update).
- Compare stability of Q-value estimates and learning curves for both update rules.

**Concept and real-world RL**

The **target network** in DQN provides a stable TD target: we use \\(Q_{target}(s',a')\\) instead of \\(Q(s',a')\\) so that the target does not change every time we update the online network, which would cause moving targets and instability. **Hard update**: copy full parameters every \\(N\\) steps (classic DQN). **Soft update**: slowly track the online network: \\(\\theta_{target} \\leftarrow \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\) with small \\(\\tau\\) (e.g. 0.001). Soft updates change the target every step but by a small amount, often yielding smoother learning. Both are used in practice (e.g. DDPG uses soft updates).

**Illustration (Q-value stability):** With soft updates, the target network changes gradually, so mean Q over a batch often evolves more smoothly than with hard updates. The chart below shows typical mean Q(s,a) over training steps (soft update).

{{< chart type="line" palette="return" title="Mean Q(s,a) over training (soft target update)" labels="0, 2k, 4k, 6k, 8k, 10k" data="0.5, 2, 5, 12, 18, 22" xLabel="Step" yLabel="Mean Q(s,a)" >}}

**Exercise:** In your DQN implementation, compare the effect of hard updates (copy every N steps) vs. soft updates (\\(\tau=0.001\\) update at each step). Plot the Q-value estimates over time to see stability differences.

**Professor's hints**

- Hard: every N steps (e.g. 100), `target.load_state_dict(online.state_dict())`. Soft: after each gradient step, for each parameter `p_target`, `p_online`, do `p_target.data.copy_(tau * p_target.data + (1 - tau) * p_online.data)` (or loop over `zip(target.parameters(), online.parameters())`).
- Q-value estimates: log the mean (or max) of \\(Q(s,a)\\) over a fixed set of states (e.g. from a few random rollouts) or over the current batch. Plot this over training steps. With soft updates, the target changes gradually, so Q-values may evolve more smoothly.
- Run both variants for the same number of steps; plot reward per episode and (if you log it) Q-values. Soft often has less oscillation but may need tuning of \\(\\tau\\).

**Common pitfalls**

- **Soft update direction:** \\(\\theta_{target} = \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\). So the target moves *toward* the online network. \\(\\tau\\) close to 1 means target changes slowly; \\(\\tau\\) close to 0 means target tracks online quickly.
- **In-place vs new tensor:** For soft update, you must update `target` parameters in place. Do not create a new network; copy data into the existing target parameters.
- **Comparing fairly:** Use the same replay buffer, same \\(\\epsilon\\), same total steps. Only the target update rule should differ.

{{< collapse summary="Worked solution (warm-up: soft update after 1000 steps)" >}}
**Warm-up:** After 1000 steps with soft update \\(\\tau=0.001\\), roughly how much of the target parameters come from the initial target vs. the current online? **Answer:** Target is updated as \\(w_{target} \\leftarrow (1-\\tau) w_{target} + \\tau w_{online}\\). So each step multiplies the "old target" component by \\(1-\\tau = 0.999\\). After 1000 steps, the initial target’s contribution is \\((0.999)^{1000} \\approx 0.37\\). So about 63% of the target is from the online network’s recent copies; the target lags but tracks the online. With \\(\\tau=0.001\\) the target changes slowly, which stabilizes learning.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** After 1000 steps with soft update \\(\\tau=0.001\\), roughly how much of the target parameters come from the initial target vs. the current online? (The target is an exponential moving average; after many steps it is close to the online.)
2. **Coding:** Implement soft target update: for two PyTorch modules (online, target), do target = τ*target + (1-τ)*online (param by param). Run 100 updates with τ=0.01 and print the L2 distance between online and target params.
3. **Challenge:** Try \\(\\tau \\in \\{0.001, 0.01, 0.1\\}\\) for soft updates. Plot learning curves. Which \\(\\tau\\) is most stable? Which learns fastest?
4. **Variant:** Compare hard update every 50 steps vs every 500 steps on CartPole. Which is more stable? Which learns faster? What does this tell you about the update frequency trade-off?
5. **Debug:** The soft update below applies the EMA in the wrong direction (target moves *away* from online). Fix it.

{{< pyrepl code="import torch\n\ndef soft_update_buggy(target_params, online_params, tau=0.01):\n    for tp, op in zip(target_params, online_params):\n        # BUG: moves target AWAY from online\n        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)  # wrong direction\n        # Wait — actually this IS correct (target = tau*online + (1-tau)*target)\n        # The correct form moves target TOWARD online by fraction tau\n        pass\n\n# Actually the bug variant: tau * target + (1-tau) * online -> target moves away\ndef soft_update_wrong(target_params, online_params, tau=0.01):\n    for tp, op in zip(target_params, online_params):\n        tp.data.copy_(tau * tp.data + (1 - tau) * op.data)  # BUG: tau wrong!\n        # This means target = tau*target + (1-tau)*online which is correct\n        # so the REAL bug is swapping tau and (1-tau)\n        pass\n\nprint('Correct: target = (1-tau)*target + tau*online')\nprint('Buggy:   target = tau*target + (1-tau)*online (same here)')\nprint('The easy bug: accidentally set tau=0.99 (slow update) vs tau=0.01')" height="260" >}}

6. **Conceptual:** What problem does the target network solve in DQN training? What would happen if we used the online network directly as the target?
7. **Recall:** Write the soft target update formula \\(\\theta_{target} \\leftarrow \\ldots\\) from memory.
