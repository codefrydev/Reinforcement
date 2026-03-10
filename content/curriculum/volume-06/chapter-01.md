---
title: "Chapter 51: Model-Free vs. Model-Based RL"
description: "Compare Dreamer and PPO sample efficiency on Walker."
date: 2026-03-10T00:00:00Z
weight: 51
draft: false
tags: ["model-free", "model-based", "Dreamer", "PPO", "sample efficiency", "curriculum"]
keywords: ["model-free vs model-based", "Dreamer", "PPO", "Walker", "sample efficiency"]
---

**Learning objectives**

- Compare **model-free** (e.g. PPO) and **model-based** (e.g. Dreamer) RL in terms of **sample efficiency** on a continuous control task like Walker.
- Explain why model-based methods can achieve more reward per real environment step (use of imagined rollouts).
- Identify trade-offs: model bias, computation, and implementation complexity.

**Concept and real-world RL**

**Model-free** methods learn a policy or value function directly from experience; **model-based** methods learn a dynamics model and use it for planning or imagined rollouts. Model-based RL can be more **sample-efficient** because each real transition can be reused many times in the model (short rollouts, planning). In **robot navigation** and **trading**, where real data is expensive, sample efficiency matters; in **game AI**, model-based methods (e.g. MuZero) combine learning and planning. The downside is **model error** (compounding over long rollouts) and extra computation.

**Where you see this in practice:** Dreamer, MBPO, and MuZero are used in benchmarks; PPO/SAC remain standard when simplicity and robustness matter.

**Exercise:** Compare the sample efficiency of a model-based method (e.g., Dreamer) and a model-free method (e.g., PPO) on a task like Walker. Explain why model-based methods can be more sample-efficient.

**Professor's hints**

- Run both for the same number of **real env steps** (e.g. 100k). Plot return vs steps. Dreamer typically uses many model rollouts per real step; PPO uses only real data.
- Sample efficiency: which method reaches a given return (e.g. 500) in fewer real steps? Model-based can do better by learning from imagined data.
- Explain: the model generates synthetic transitions, so the agent effectively gets more "experience" per real sample.

**Common pitfalls**

- **Comparing by wall-clock time:** Model-based often does more compute per step; compare by env steps (or report both).
- **Different hyperparameters:** Use reasonable defaults for each; document so the comparison is fair.

{{< collapse summary="Worked solution (warm-up: why a learned model helps)" >}}
**Warm-up:** A learned model lets us generate *simulated* transitions without touching the real environment. We can do many updates (planning) per real step, so we extract more learning signal from each sample. That improves sample efficiency: we need fewer real env steps to reach the same performance. The trade-off is model error (compounding over long rollouts) and extra compute per step.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why can a learned model improve sample efficiency?
2. **Coding:** Run PPO on Walker2d for 200k steps and Dreamer (or MBPO) for 200k steps. Plot mean return vs steps. Which reaches 1000 first?
3. **Challenge:** Vary the model rollout length in the model-based method. How does very long rollout length affect performance (think compounding error)?
