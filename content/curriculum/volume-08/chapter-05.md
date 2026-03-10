---
title: "Chapter 75: Limitations of Behavioral Cloning"
description: "Covariate shift; DAgger: mix expert and BC, retrain."
date: 2026-03-10T00:00:00Z
weight: 75
draft: false
tags: ["DAgger", "behavioral cloning", "covariate shift", "curriculum"]
keywords: ["DAgger", "covariate shift", "expert and BC", "behavioral cloning limits"]
---

**Learning objectives**

- **Demonstrate** the **covariate shift** problem: run the BC agent, record states it visits that were rare or absent in the expert data, and show that errors compound in those regions.
- **Implement DAgger:** collect new data by running the current BC policy (or a mix of expert and BC), query the expert for the correct action at those states, add to the dataset, and retrain BC.
- **Explain** why DAgger reduces covariate shift by adding on-policy (or mixed) states to the training set.
- **Compare** BC (trained only on expert data) with DAgger (iteratively aggregated) in terms of evaluation return and robustness.
- **Relate** covariate shift and DAgger to **robot navigation** and **healthcare** where the learner's distribution can drift from the expert's.

**Concept and real-world RL**

**Covariate shift** in imitation learning occurs when the **state distribution** of the learner (BC policy) at test time differs from the **state distribution** of the expert in the training data. The BC policy was only trained on expert-visited states; when it makes a small mistake and enters a state the expert rarely visited, it has no good training signal and can make more mistakes, leading to compounding errors. **DAgger** (Dataset Aggregation) addresses this: iteratively run the current policy, get expert labels (actions) for the visited states, add them to the dataset, and retrain. This way the dataset comes to include "learner" states, so BC learns how to recover. In **robot navigation** and **healthcare**, the expert may not visit every possible state; DAgger-style data collection improves robustness.

**Where you see this in practice:** DAgger and variants; interactive imitation learning; reducing distribution shift in LfD.

**Illustration (DAgger iteration):** DAgger collects states from the current policy and retrains with expert labels; performance typically improves over iterations. The chart below shows BC vs DAgger return.

{{< chart type="line" title="Return over DAgger iterations" labels="BC only, Iter 1, Iter 2, Iter 3" data="120, 180, 220, 245" >}}

**Exercise:** In the same setting, show the covariate shift problem by letting the BC agent run and recording states it visits but never saw in the expert data. Implement DAgger: collect new data by mixing expert and BC actions, then retrain.

**Professor's hints**

- **Showing covariate shift:** Run the BC agent for 100 episodes; for each state s visited, check if s (or a discretized/binned version) appeared in the expert dataset. Plot the fraction of "novel" states per episode or over time; as the agent drifts, it may enter more novel states and then fail.
- **DAgger loop:** (1) Start with expert dataset D. (2) Train BC on D. (3) Run the BC policy (or mix: e.g. 50% expert, 50% BC) to collect trajectories. (4) For each state in these trajectories, get the **expert** action (run expert policy, or use a scripted expert). (5) Add (s, a_expert) to D. (6) Go to (2). Repeat for 3–5 iterations.
- **Mixing expert and BC:** When collecting in step (3), you can use the current BC policy, or with probability β use the expert (so some trajectories stay close to expert). β=0 is pure BC rollout; β=1 is pure expert (no new states). Try β=0.5.
- Compare final evaluation return: BC only (one train) vs DAgger after 3 iterations.

**Common pitfalls**

- **Expert action at "bad" states:** In DAgger we need the expert's action at states the BC agent visits. If the expert is a policy, run it from that state (or from the start and stop at that state). Ensure the expert is still available (e.g. saved policy) to label new states.
- **Overfitting to last iteration:** If you retrain from scratch each time, the dataset grows; if you only train on new data, you forget the old. Standard DAgger aggregates all data and retrains on the full D each time.
- **Computational cost:** DAgger requires multiple rounds of data collection and retraining; use a small number of iterations and a small env for the exercise.

{{< collapse summary="Worked solution (warm-up: DAgger)" >}}
**Key idea:** DAgger: (1) Train BC on expert data. (2) Run the current policy in the env; when it visits a state, query the expert for the action. (3) Add (agent state, expert action) to the dataset. (4) Retrain BC on the aggregated data. Repeat. So the dataset now includes states the *agent* visits, not just the expert; this reduces covariate shift and often improves over BC.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence, why does adding "states that the BC agent visits" to the training set help?
2. **Coding:** On LunarLander, train BC on 50 expert episodes. Then run 20 episodes of the BC policy and record all (s, a_BC). For each s, get a_expert from the expert policy. Add these to the dataset and retrain BC. Repeat once more. Plot evaluation return: initial BC, after 1 DAgger round, after 2 rounds.
3. **Challenge:** Implement **stochastic mixing**: at each step during data collection, with probability β take the expert action, else take BC action. Vary β and see how it affects the diversity of states and final BC performance after 3 DAgger iterations.
