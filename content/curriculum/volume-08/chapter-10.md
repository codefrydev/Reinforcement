---
title: "Chapter 80: RL from Human Feedback (RLHF) Basics"
description: "Bradley-Terry from pairwise comparisons; train policy with PPO."
date: 2026-03-10T00:00:00Z
weight: 80
draft: false
tags: ["RLHF", "Bradley-Terry", "preferences", "PPO", "curriculum"]
keywords: ["RLHF", "human feedback", "Bradley-Terry", "pairwise comparisons", "PPO"]
---

**Learning objectives**

- **Implement** a **Bradley-Terry model** to learn a reward function from **pairwise comparisons** of two trajectories (or segments): given (τ^w, τ^l) meaning "τ^w is preferred over τ^l," fit r so that E[r(τ^w)] > E[r(τ^l)].
- **Use** the learned reward to **train a policy with PPO** (or another policy gradient method): maximize expected return under r.
- **Explain** the RLHF pipeline: collect preferences → train reward model → train policy on reward model.
- **Test** on a simple environment with simulated preferences (e.g. prefer longer/higher-return trajectories) and verify the policy improves.
- **Relate** RLHF to **dialogue** (prefer helpful/harmless responses) and **recommendation** (prefer engaging content).

**Concept and real-world RL**

**Reinforcement learning from human feedback (RLHF)** learns a **reward function** from **human preferences** (e.g. "response A is better than response B") and then trains a policy to maximize that reward. The **Bradley-Terry model** assumes the probability that trajectory τ^w is preferred over τ^l is σ(r(τ^w) - r(τ^l)), where r(τ) = sum_t r(s_t, a_t) and σ is the logistic function. We fit r (e.g. a neural network) by maximizing the likelihood of the observed comparisons. Then we run PPO (or similar) with the learned reward. In **dialogue** and **recommendation**, humans cannot easily write a reward function but can compare outcomes; RLHF bridges that gap.

**Where you see this in practice:** RLHF for language models (ChatGPT, Claude); preference-based RL; learning reward from comparisons.

**Illustration (Bradley-Terry reward):** From pairwise preferences we learn a reward model; PPO with that reward improves. The chart below shows PPO return and correlation between learned reward and true return over training.

{{< chart type="line" palette="return" title="PPO return (reward from preferences)" labels="0, 50, 100, 150, 200" data="100, 180, 220, 245, 255" xLabel="Iteration" yLabel="Return" >}}

**Exercise:** Given a set of pairwise comparisons of two trajectories, implement a Bradley-Terry model to learn a reward function. Then use that reward to train a policy with PPO.

**Professor's hints**

- **Bradley-Terry:** For each comparison (τ^w, τ^l), loss = -log σ(r(τ^w) - r(τ^l)). Here r(τ) = sum over t of r(s_t, a_t). Use a reward network r_ψ(s, a) or r_ψ(s); then r(τ) = sum_t r_ψ(s_t, a_t). Minimize negative log likelihood over the comparison dataset.
- **Reward network:** Input state (and optionally action); output scalar. Train on comparisons; ensure rewards are normalized or scaled (e.g. baseline subtraction) so that PPO gets a reasonable signal.
- **PPO phase:** Freeze the reward network; use R(t) = sum_{t'>=t} γ^{t'-t} r_ψ(s_{t'}, a_{t'}) as the return for the policy. Run PPO as usual. Optionally add a KL penalty to prevent the policy from drifting too far from the initial policy (as in full RLHF for LLMs).
- **Simulated preferences:** For testing, create comparisons from a simple rule (e.g. prefer trajectory with higher true return, or longer trajectory). Then you can verify that the learned reward correlates with the true objective and that PPO improves.

**Common pitfalls**

- **Reward hacking:** The policy may find behaviors that get high learned reward but do not match human intent (e.g. short trajectories that get high per-step reward). Use a KL penalty or limit the number of PPO steps.
- **Sparse comparisons:** If you have few comparisons, the reward model may be underdetermined; use regularization or a prior (e.g. small network, weight decay).
- **Bias in preferences:** If the preference data is biased (e.g. always preferring longer responses), the reward will reflect that. For the exercise, simulated preferences are fine; for real RLHF, data quality matters.

{{< collapse summary="Worked solution (warm-up: RLHF)" >}}
**Key idea:** RLHF: (1) Collect preference data (A vs B; which is better?). (2) Train a reward model \\(r_\\psi\\) to predict the preference (e.g. Bradley-Terry: \\(P(A>B) = \\sigma(r(A)-r(B))\\)). (3) Use RL (e.g. PPO) to maximize \\(r_\\psi\\) while staying close to a reference policy (KL penalty) to avoid reward hacking. So we learn human preferences as a reward and optimize it with RL. Used for LLM alignment.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why do we use pairwise comparisons instead of asking humans to assign a numeric reward to each trajectory?
2. **Coding:** Generate 1000 pairs of trajectories from a random policy on CartPole. Label each pair: prefer the trajectory with higher true return. Train a reward network with Bradley-Terry. Then train PPO with this reward for 200 iterations. Plot PPO return and correlation between learned r(τ) and true return.
3. **Challenge:** Add a **KL penalty** to PPO: penalize π(a|s) deviating from π_0(a|s) (initial policy). This is standard in RLHF to prevent mode collapse. Compare policy behavior and return with and without KL penalty.
4. **Variant:** Reduce the number of human comparisons from 1000 to 100. How much does the learned reward quality degrade (measure by correlation with true return)? At what sample count does the reward model become unreliable?
5. **Debug:** The reward model achieves 90% accuracy on held-out comparison pairs, but PPO trained with this reward quickly diverges to a degenerate policy (e.g. always spinning in place). The KL penalty is set to β=0. Explain reward hacking: the policy found a behavior that maximizes the learned reward but violates the intent, and why a KL penalty or reward model ensemble variance penalty would help.
6. **Conceptual:** Human preference labels in RLHF can be inconsistent, biased, or context-dependent. Describe two failure modes of RLHF that arise from annotator bias, and explain what makes reward hacking particularly hard to detect in language model settings where the output space is vast.
