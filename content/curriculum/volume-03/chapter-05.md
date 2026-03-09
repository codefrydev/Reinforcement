---
title: "Chapter 25: Target Networks"
description: "Hard vs soft target updates in DQN."
date: 2026-03-10T00:00:00Z
weight: 25
draft: false
---

**Learning objectives**

- Implement **hard** target updates: copy online network parameters to the target network every \\(N\\) steps.
- Implement **soft** target updates: \\(\\theta_{target} \\leftarrow \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\) each step (or each update).
- Compare stability of Q-value estimates and learning curves for both update rules.

**Concept and real-world RL**

The **target network** in DQN provides a stable TD target: we use \\(Q_{target}(s',a')\\) instead of \\(Q(s',a')\\) so that the target does not change every time we update the online network, which would cause moving targets and instability. **Hard update**: copy full parameters every \\(N\\) steps (classic DQN). **Soft update**: slowly track the online network: \\(\\theta_{target} \\leftarrow \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\) with small \\(\\tau\\) (e.g. 0.001). Soft updates change the target every step but by a small amount, often yielding smoother learning. Both are used in practice (e.g. DDPG uses soft updates).

**Exercise:** In your DQN implementation, compare the effect of hard updates (copy every N steps) vs. soft updates (\\(\tau=0.001\\) update at each step). Plot the Q-value estimates over time to see stability differences.

**Professor's hints**

- Hard: every N steps (e.g. 100), `target.load_state_dict(online.state_dict())`. Soft: after each gradient step, for each parameter `p_target`, `p_online`, do `p_target.data.copy_(tau * p_target.data + (1 - tau) * p_online.data)` (or loop over `zip(target.parameters(), online.parameters())`).
- Q-value estimates: log the mean (or max) of \\(Q(s,a)\\) over a fixed set of states (e.g. from a few random rollouts) or over the current batch. Plot this over training steps. With soft updates, the target changes gradually, so Q-values may evolve more smoothly.
- Run both variants for the same number of steps; plot reward per episode and (if you log it) Q-values. Soft often has less oscillation but may need tuning of \\(\\tau\\).

**Common pitfalls**

- **Soft update direction:** \\(\\theta_{target} = \\tau \\theta_{target} + (1-\\tau) \\theta_{online}\\). So the target moves *toward* the online network. \\(\\tau\\) close to 1 means target changes slowly; \\(\\tau\\) close to 0 means target tracks online quickly.
- **In-place vs new tensor:** For soft update, you must update `target` parameters in place. Do not create a new network; copy data into the existing target parameters.
- **Comparing fairly:** Use the same replay buffer, same \\(\\epsilon\\), same total steps. Only the target update rule should differ.

**Extra practice**

1. **Warm-up:** After 1000 steps with soft update \\(\\tau=0.001\\), roughly how much of the target parameters come from the initial target vs. the current online? (The target is an exponential moving average; after many steps it is close to the online.)
2. **Challenge:** Try \\(\\tau \\in \\{0.001, 0.01, 0.1\\}\\) for soft updates. Plot learning curves. Which \\(\\tau\\) is most stable? Which learns fastest?
