---
title: "Chapter 30: Rainbow DQN"
description: "Combine DDQN, Dueling, PER, NoisyNet, multi-step; train on Pong."
date: 2026-03-10T00:00:00Z
weight: 30
draft: false
tags: ["Rainbow", "DQN", "Pong", "DDQN", "Dueling", "PER", "curriculum"]
keywords: ["Rainbow DQN", "DDQN Dueling PER NoisyNet", "Pong", "combined improvements"]
---

**Learning objectives**

- Combine **Rainbow** components: Double DQN, Dueling architecture, Prioritized replay, Noisy networks, and optionally multi-step returns (and distributional RL).
- Train on a challenging environment (e.g. Pong or another Atari-style env) and compare with a baseline DQN.
- Understand which components contribute most to sample efficiency and stability.

**Concept and real-world RL**

**Rainbow** (Hessel et al.) combines several DQN improvements: Double DQN (reduce overestimation), Dueling (value + advantage), PER (replay important transitions), Noisy nets (state-dependent exploration), multi-step returns (n-step learning), and optionally C51 (distributional RL). Together they improve sample efficiency and final performance on Atari. In practice, you do not need all components for every task; CartPole may be solved with vanilla DQN, while harder games benefit from the full stack. Implementing Rainbow is a capstone for the value-approximation volume.

**Illustration (Rainbow vs DQN):** With the same number of env steps, Rainbow often reaches higher reward sooner. The chart below shows mean reward per 100 episodes over 1M steps (e.g. Pong or LunarLander).

{{< chart type="line" palette="return" title="Mean reward per 100 episodes (Rainbow vs DQN)" labels="0, 200k, 400k, 600k, 800k, 1M" data="-20, 0, 5, 12, 18, 20" xLabel="Step" yLabel="Mean reward" >}}

**Exercise:** Combine all improvements (DDQN, Dueling, PER, NoisyNet, multi-step returns, distributional RL optional) into a single Rainbow agent. Train it on a challenging environment like Pong and compare with a baseline DQN.

**Professor's hints**

- Start from your best DQN (e.g. with target network and replay). Add one component at a time: first DDQN, then Dueling, then PER (with IS weights), then Noisy layers (replace \\(\\epsilon\\)-greedy). Multi-step: use n-step returns in the target (e.g. 3-step); you will need to store n-step transitions or compute targets over n steps.
- Pong (or Atari) needs frame stacking (e.g. 4 frames) and possibly frame skip. Use a CNN if the observation is image-based. For a simpler "challenging" env, use LunarLander or a harder CartPole variant.
- Comparison: same number of env steps (e.g. 1M). Plot reward per episode (or per 100 episodes). Rainbow should reach higher performance and/or learn faster. Ablation: remove one component and see how much performance drops.

**Common pitfalls**

- **Overfitting the baseline:** Use a strong baseline (e.g. DQN with replay + target + maybe DDQN). A very weak baseline makes Rainbow look good even if the gain is from one component.
- **Hyperparameters:** Rainbow has more hyperparameters (PER \\(\\alpha\\), \\(\\beta\\), Noisy init, n-step). Tune or use published defaults; do not compare with a heavily tuned DQN vs. untuned Rainbow.
- **Distributional (C51):** Optional and more complex (output a distribution over returns per action, then project and minimize cross-entropy). You can skip it and still have a "Rainbow-lite" that is very effective.

{{< collapse summary="Worked solution (warm-up: six components of Rainbow)" >}}
**Warm-up:** List the six (or seven) components of Rainbow. For each, state in one sentence what problem it addresses. **Answer:** (1) **DQN** — base: replay + target network for stability. (2) **Double DQN** — reduces max overestimation. (3) **Prioritized replay** — samples important transitions more. (4) **Dueling** — separates V and A for better learning when actions don’t matter much. (5) **Multi-step (n-step)** — uses n-step returns for less bias. (6) **Noisy nets** — parameter space exploration instead of \\(\\epsilon\\)-greedy. (7) **C51 (distributional)** — learns return distribution. Together they improve sample efficiency and stability.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** List the six (or seven) components of Rainbow. For each, state in one sentence what problem it addresses.
2. **Coding:** Implement a minimal "Rainbow-lite": DQN + replay + target + Double DQN + Dueling. Train on CartPole for 20k steps. Log mean Q and episode return.
3. **Challenge:** Ablation study: train Rainbow, then train variants with each component removed (Rainbow - DDQN, Rainbow - Dueling, etc.). Rank the components by how much removing them hurts performance.
4. **Variant:** Add n-step returns (n=3) to your Rainbow-lite. Does multi-step help or hurt on CartPole? Try n=1 vs n=3 and compare learning speed.
5. **Debug:** The target computation below uses \\(\\max\\) of two Q-networks instead of \\(\\min\\), reintroducing DDPG-style overestimation. Fix it.

{{< pyrepl code="import torch\n\ndef compute_target(Q1_target, Q2_target, actions, rewards, dones, gamma=0.99):\n    with torch.no_grad():\n        # BUG: should use min(Q1, Q2), not max\n        next_q = torch.max(Q1_target, Q2_target)  # wrong\n        target = rewards + gamma * (1 - dones) * next_q.max(dim=1).values\n    return target\n\n# Fix: next_q = torch.min(Q1_target, Q2_target)\nQ1 = torch.tensor([[0.5, 0.9]])\nQ2 = torch.tensor([[0.8, 0.6]])\nprint('Buggy next_q:', torch.max(Q1, Q2))\nprint('Correct next_q:', torch.min(Q1, Q2))" height="220" >}}

6. **Conceptual:** Which single Rainbow component typically provides the largest single improvement over vanilla DQN on Atari, and why? (Consider PER, Double DQN, and Dueling separately.)
7. **Recall:** State in 2–3 sentences what Rainbow is: which paper introduced it, what components it combines, and what benchmark it was evaluated on.
