---
title: "Chapter 17: Planning and Learning with Tabular Methods"
description: "Dyna-Q on 4×4 deterministic gridworld."
date: 2026-03-10T00:00:00Z
weight: 17
draft: false
tags: ["Dyna-Q", "planning", "tabular", "gridworld", "curriculum"]
keywords: ["Dyna-Q", "planning and learning", "tabular methods", "model-based"]
---

**Learning objectives**

- Implement a simple **model**: store \\((s,a) \\rightarrow (r, s')\\) from experience.
- Implement **Dyna-Q**: after each real env step, do \\(k\\) extra Q-updates using random \\((s,a)\\) from the model (simulated experience).
- Compare sample efficiency: Dyna-Q (planning + learning) vs Q-learning (learning only).

**Concept and real-world RL**

**Model-based** methods use a learned or given model of the environment (transition and reward). **Dyna-Q** learns a tabular model from real experience: when you observe \\((s,a,r,s')\\), store it. Then, in addition to updating \\(Q(s,a)\\) from the real transition, you *replay* random \\((s,a)\\) from the model, look up \\((r,s')\\), and do a Q-learning update. This gives more learning per real step (planning). In real applications, learned models are used in model-based RL (e.g. world models, MuZero) to reduce sample complexity; the key idea is reusing past experience for extra updates.

**Illustration (sample efficiency):** Dyna-Q does multiple Q-updates per real env step (e.g. 1 real + 5 planning). Cumulative reward often rises faster than with Q-learning alone. The chart below shows cumulative reward over the first 200 real steps.

{{< chart type="line" palette="return" title="Cumulative reward over real steps (Dyna-Q vs Q-learning)" labels="0, 50, 100, 150, 200" data="-100, -60, -25, 10, 50" xLabel="Step" yLabel="Cumulative reward" >}}

**Exercise:** Implement Dyna-Q on a simple deterministic gridworld (4×4). Use a model that stores observed transitions. After each real step, perform 5 planning updates using randomly sampled state-action pairs from the model. Compare with Q-learning without planning.

**Professor's hints**

- Model: a dict mapping \\((s,a)\\) to \\((r, s')\\). For deterministic env, one transition per \\((s,a)\\). When you see \\((s,a,r,s')\\), set `model[(s,a)] = (r, s')`.
- After each real env step: (1) update \\(Q(s,a)\\) with the real \\((r,s')\\); (2) add \\((s,a,r,s')\\) to the model; (3) sample 5 random \\((s,a)\\) from the model (e.g. from a list of keys), get \\((r,s')\\), and do 5 Q-learning updates.
- Comparison: run Q-learning and Dyna-Q for the same number of *real* steps (e.g. 500). Plot cumulative reward or success rate. Dyna-Q should often reach good performance in fewer real steps because of the planning updates.

**Common pitfalls**

- **Sampling only recently seen (s,a):** The model can store *all* observed \\((s,a)\\); sample uniformly from the model (or from a list of keys). If you only sample the last transition, planning is weak.
- **Deterministic model:** In a deterministic gridworld, each \\((s,a)\\) has one \\((r,s')\\). In stochastic envs you would store a distribution or multiple samples; for this exercise deterministic is fine.
- **Q-learning update in planning:** Use the same update rule: \\(Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]\\) with the \\((r,s')\\) from the model.

{{< collapse summary="Worked solution (warm-up: Dyna-Q update count)" >}}
**Warm-up:** After 100 real steps, how many (s,a) pairs might your model contain? How many total Q-updates does Dyna-Q do? **Answer:** The model stores at most one entry per (s,a) *observed*; after 100 steps you have at most 100 (s,a) pairs (fewer if repeated). Total Q-updates = 100 (one per real step) + 100 × 5 = 600 (100 real + 500 planning). So Dyna-Q does 6× more updates per real step; that’s why it can learn faster with the same number of env interactions.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** After 100 real steps, how many (s,a) pairs might your model contain? How many total Q-updates does Dyna-Q do in those 100 steps (real + planning)?
2. **Coding:** Implement a simple deterministic model (dict (s,a) -> (s', r)) and Dyna-Q: after each real step, do k=5 planning steps (sample random (s,a) from model, update Q). Compare with Q-learning (k=0) on a small gridworld.
3. **Challenge:** Vary the number of planning steps \\(k \\in \\{0, 5, 20\\}\\). Plot learning curves. Does more planning always help? What if \\(k\\) is very large?
