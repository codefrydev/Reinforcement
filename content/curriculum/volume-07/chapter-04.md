---
title: "Chapter 64: Random Network Distillation (RND)"
description: "RND: fixed target, predictor; prediction error as intrinsic reward."
date: 2026-03-10T00:00:00Z
weight: 64
draft: false
tags: ["RND", "random network distillation", "intrinsic reward", "curriculum"]
keywords: ["RND", "random network distillation", "prediction error", "intrinsic reward"]
---

**Learning objectives**

- **Implement** RND: a fixed random target network and a predictor network that fits the target on visited states.
- **Use** prediction error (target output vs predictor output) as intrinsic reward for exploration.
- **Explain** why RND rewards novelty without learning a forward model of the environment.
- **Apply** RND to a hard exploration problem (e.g. Pitfall-style or sparse-reward maze) and compare with ε-greedy or count-based exploration.
- **Relate** RND to **game AI** and **robot navigation** where state spaces are large and rewards sparse.

**Concept and real-world RL**

**Random Network Distillation (RND)** provides an intrinsic reward by comparing the output of a **fixed random network** (target) with a **predictor network** trained to match it on visited states. States that have been visited often have low prediction error; novel states have high error and thus high intrinsic reward. Unlike ICM, RND does not use actions or a forward model—it only measures "how well can we predict this state's fingerprint?" In **game AI** (e.g. Atari hard-exploration games like Pitfall), RND helps agents discover new screens; in **robot navigation**, it can encourage visiting new regions of the state space when the goal reward is sparse.

**Where you see this in practice:** RND in Atari (e.g. OpenAI); bonus-based exploration in DQN and policy-gradient methods.

**Illustration (RND exploration):** RND intrinsic reward is high in novel states (high prediction error). The chart below shows extrinsic vs intrinsic return over episodes (intrinsic decreases as states become familiar).

{{< chart type="line" palette="return" title="Intrinsic return over episodes (RND)" labels="0, 50, 100, 150, 200" data="80, 50, 30, 18, 10" xLabel="Episode" yLabel="Intrinsic return" >}}

**Exercise:** Implement RND: a fixed random target network and a predictor network trained on visited states. Use the prediction error as intrinsic reward. Apply it to a hard exploration problem like a Pitfall-style environment.

**Professor's hints**

- **Target:** One or more layers of a randomly initialized network (e.g. MLP or small CNN) that maps state to a fixed-dimensional vector; weights are frozen.
- **Predictor:** Same architecture (or similar), trained to match the target's output on the current batch of states. Loss = MSE(target(s), predictor(s)); intrinsic reward = MSE (or its square root) so that novel states get high reward.
- If Pitfall-style env is not available, use a **deterministic maze** with a single distant goal or a **proc-gen maze** where each episode has a different layout so that "novelty" is meaningful.
- Add the intrinsic reward to the environment reward (with a coefficient); tune so the agent both explores and still cares about the goal.

**Common pitfalls**

- **Predictor overtrains on recent states:** If the predictor fits the replay buffer too well, intrinsic reward drops everywhere; use a reasonable buffer size and learning rate so the predictor generalizes to "similar" states but fails on truly new ones.
- **Scaling:** Raw MSE can be large or small depending on architecture; normalize or scale the intrinsic reward so it is on a similar scale to extrinsic reward.
- **Continuous state spaces:** RND works well with raw pixels or state vectors; for high-dimensional states, the target and predictor need enough capacity to make "novel" states distinguishable.

{{< collapse summary="Worked solution (warm-up: RND)" >}}
**Key idea:** RND: a random fixed network maps state to a target; a predictor network is trained to match the target. Intrinsic reward = \\(\\|f_{target}(s) - f_{pred}(s)\\|^2\\). States where the predictor fails (high error) are "novel." The target is random so it does not change; only the predictor is trained. This gives a stable novelty signal without count-based memory.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does a fixed random network give different outputs for different inputs? Why would a predictor trained on visited states have higher error on unvisited states?
2. **Coding:** Implement RND on a 10×10 maze with goal at (9,9). Plot "mean intrinsic reward" and "unique states visited" over training. Compare with the same agent without RND.
3. **Challenge:** Combine RND with **count-based** exploration (e.g. use both RND bonus and \\(1/\\sqrt{N(s)}\\)). Does the combination improve over either alone on a hard exploration task?
4. **Variant:** Change the target network architecture (small MLP vs larger MLP vs CNN for pixel inputs). Does the capacity of the target network affect exploration quality? Run on a pixel-based env and compare coverage.
5. **Debug:** An RND agent's intrinsic reward stays near zero after the first 1000 steps even in unvisited states. Logging shows the predictor network parameters are identical to the target network's parameters. Identify the bug (both networks were initialized with the same seed and neither is frozen) and fix it.
6. **Conceptual:** RND produces *episodic* vs *lifetime* curiosity depending on whether the predictor is reset between episodes. Explain the difference: when would you want the predictor to persist across episodes, and when would you reset it?
