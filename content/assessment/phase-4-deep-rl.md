---
title: "Phase 4 Deep RL Quiz"
description: "10–12 questions on DQN, policy gradient, PPO, replay, target network. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["assessment", "phase 4", "deep RL", "DQN", "PPO", "policy gradient", "solutions"]
keywords: ["phase 4 deep RL", "DQN", "policy gradient", "PPO", "replay target network", "solutions", "deep RL quiz"]
---

Use this quiz after completing [Volumes 3–5](../curriculum/) (or the [Phase 4 coding challenges](../learning-path/phase-4/)). If you can answer at least 9 of 12 correctly, you are ready for [Phase 5](../learning-path/#phase-5--advanced-topics) and [Volume 6](../curriculum/volume-06/).

---

### 1. Function approximation

**Q:** Why is function approximation necessary in RL for large or continuous state spaces?

{{< collapse summary="Answer" >}}
Tabular methods store one value per state (or state-action); the number of states can be huge or infinite. Function approximation uses a parameterized function (e.g. neural network) so a fixed number of parameters represent values for all states and generalize from seen to unseen states.
{{< /collapse >}}

---

### 2. DQN target

**Q:** Write the TD target \\(y\\) for DQN given a transition \\((s, a, r, s', \\mathrm{done})\\). Why do we use a *target* network for \\(y\\)?

{{< collapse summary="Answer" >}}
**Formula:** \\(y = r + \\gamma (1 - \\mathrm{done}) \\max_{a'} Q_{target}(s', a')\\). When done=1 we have \\(y = r\\) (no bootstrap). **Why target network:** The target would otherwise depend on the same network we are updating, so it would change every step (moving target) and training would be unstable. A slowly updated copy (e.g. soft or periodic update) gives a stable label.
{{< /collapse >}}

---

### 3. Experience replay

**Q:** Why is experience replay used in DQN? What problem does it solve?

{{< collapse summary="Answer" >}}
Replay stores past transitions and samples random minibatches. It **breaks correlation** between consecutive updates (so we don't overfit to the last few transitions) and **reuses data** (sample efficiency). It also stabilizes training.
{{< /collapse >}}

---

### 4. Double DQN

**Q:** How does Double DQN differ from DQN in computing the TD target? Why does it reduce overestimation?

{{< collapse summary="Answer" >}}
Double DQN uses the **online** network to select \\(a^* = \\arg\\max_a Q_{online}(s', a)\\) and the **target** network to evaluate \\(Q_{target}(s', a^*)\\). The max over Q-values is biased high when Q is noisy; decoupling selection and evaluation reduces this overestimation.
{{< /collapse >}}

---

### 5. Policy gradient

**Q:** Write the policy gradient update (gradient ascent) for expected return \\(J(\\theta)\\). What quantity do we use in place of the full return in practice (e.g. in REINFORCE)?

{{< collapse summary="Answer" >}}
**Update:** \\(\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J\\) (gradient ascent because we maximize return). **In practice:** We use a sample: \\(\\nabla \\log \\pi(a_t|s_t) \\cdot G_t\\) (REINFORCE) or with a baseline \\(G_t - b(s_t)\\). The return \\(G_t\\) (or advantage) weights the gradient so we increase the probability of actions that led to high return. **Why log:** The policy gradient theorem expresses \\(\\nabla J\\) as an expectation of \\(\\nabla \\log \\pi \\cdot\\) (return or advantage).
{{< /collapse >}}

---

### 6. Baseline

**Q:** Why does subtracting a state-dependent baseline from the return in the policy gradient keep the gradient unbiased?

{{< collapse summary="Answer" >}}
\\(\\mathbb{E}[ \\nabla \\log \\pi(a|s) \\cdot b(s) ] = b(s) \\mathbb{E}[ \\nabla \\log \\pi(a|s) ] = b(s) \\cdot 0 = 0\\) (the expected gradient of log-probability is zero). So subtracting \\(b(s)\\) does not change the expectation of the gradient but can reduce variance.
{{< /collapse >}}

---

### 7. Actor-critic

**Q:** What is the advantage of actor-critic over REINFORCE? What do we use as the advantage in the simplest actor-critic?

{{< collapse summary="Answer" >}}
Actor-critic uses a **value function** (critic) to reduce the **variance** of the policy gradient estimate. The simplest advantage is the **TD error**: \\(r + \\gamma V(s') - V(s)\\) (one-step), which has lower variance than the full return \\(G_t\\).
{{< /collapse >}}

---

### 8. PPO clip

**Q:** What is the purpose of the clipped objective in PPO? Write the clipped surrogate in one line (ratio \\(r_t\\), advantage \\(\\hat{A}_t\\), clip range \\(\\epsilon\\)).

{{< collapse summary="Answer" >}}
To **prevent too large policy updates** (which can cause collapse or instability). \\(L^{CLIP} = \\min( r_t \\hat{A}_t, \\mathrm{clip}(r_t, 1-\\epsilon, 1+\\epsilon) \\hat{A}_t )\\). We maximize the expectation of this over the batch.
{{< /collapse >}}

---

### 9. GAE

**Q:** What is GAE (Generalized Advantage Estimation)? How does \\(\\lambda\\) trade off bias and variance?

{{< collapse summary="Answer" >}}
GAE is a weighted sum of TD errors: \\(\\hat{A}_t = \\sum_{l\\ge 0} (\\gamma\\lambda)^l \\delta_{t+l}\\). \\(\\lambda=0\\) gives 1-step TD (low variance, high bias); \\(\\lambda=1\\) gives Monte Carlo (high variance, low bias). \\(\\lambda \\in (0,1)\\) balances them.
{{< /collapse >}}

---

### 10. SAC

**Q:** What is the maximum entropy objective in SAC? Why does adding entropy help?

{{< collapse summary="Answer" >}}
We maximize \\(\\mathbb{E}[ \\sum_t r_t + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t)) ]\\). Entropy encourages **exploration** (more stochastic policy) and can improve robustness and sample efficiency. The temperature \\(\\alpha\\) is often auto-tuned.
{{< /collapse >}}

---

### 11. Continuous actions

**Q:** For continuous actions (e.g. Pendulum), how do we typically parameterize the policy? How do we sample and compute log-probability?

{{< collapse summary="Answer" >}}
We use a **Gaussian** policy: network outputs mean \\(\\mu(s)\\) and log-std \\(\\log \\sigma(s)\\). Sample \\(a = \\mu + \\sigma \\cdot z\\), \\(z \\sim \\mathcal{N}(0,1)\\). Log-prob: \\(\\log \\pi(a|s) = -\\frac{1}{2}(\\log(2\\pi) + 2\\log\\sigma + (a-\\mu)^2/\\sigma^2)\\). For bounded actions we may use tanh squashing with a Jacobian correction.
{{< /collapse >}}

---

### 12. DDPG vs SAC

**Q:** In one sentence each: is DDPG on-policy or off-policy? Is SAC? What exploration does each use?

{{< collapse summary="Answer" >}}
**DDPG:** off-policy (replay buffer); exploration via **action noise** (e.g. OU or Gaussian). **SAC:** off-policy (replay); exploration via **maximum entropy** (stochastic policy), no separate noise needed.
{{< /collapse >}}

---

**Next step:** If you passed, go to [Phase 5 — Advanced topics](../learning-path/#phase-5--advanced-topics) and [Volume 6](../curriculum/volume-06/).
