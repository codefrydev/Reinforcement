---
title: "Chapter 97: Direct Preference Optimization (DPO)"
description: "DPO loss from Bradley-Terry and KL-optimal policy; compare with PPO."
date: 2026-03-10T00:00:00Z
weight: 97
draft: false
tags: ["DPO", "Direct Preference Optimization", "Bradley-Terry", "PPO", "curriculum"]
keywords: ["DPO", "Direct Preference Optimization", "Bradley-Terry", "KL-optimal policy"]
---

**Learning objectives**

- **Derive** the **DPO loss** from the **Bradley-Terry** preference model and the **optimal policy** under a KL constraint to the reference policy (the closed-form mapping from reward to policy in the BT model).
- **Implement** DPO: train the language model **directly** on preference data (prefer τ^w over τ^l) using the DPO loss, **without** training a separate reward model.
- **Compare** with **PPO** (reward model + PPO fine-tuning) in terms of preference accuracy, reward model score, and implementation complexity.
- **Explain** the advantage of DPO: no reward model, no PPO loop; just supervised loss on preferences.
- **Relate** DPO to **dialogue** and **RLHF** (alternative to reward model + PPO).

**Concept and real-world RL**

**Direct Preference Optimization (DPO)** avoids the reward-model and PPO steps of RLHF. Under the **Bradley-Terry** model, there is a closed-form relationship between the optimal policy (under a KL constraint to the reference) and the reward function. DPO turns this into a **supervised loss** on preference data: we maximize the likelihood that the preferred response τ^w is ranked above τ^l under the current policy, using the DPO formula that involves only π(τ^w)/π_ref(τ^w) and π(τ^l)/π_ref(τ^l). So we train the LM directly on (prompt, τ^w, τ^l) without a reward model or PPO. In **dialogue** and **RLHF**, DPO is a simpler and often more stable alternative to PPO-based RLHF.

**Where you see this in practice:** DPO and variants (IPO, KTO); alignment without reward model; preference-based fine-tuning.

**Illustration (DPO vs PPO):** DPO trains directly on preferences without a separate reward model. The chart below shows preference accuracy (or reward) over training for DPO vs PPO-style RLHF.

{{< chart type="line" palette="return" title="Preference accuracy (DPO)" labels="0, 200, 400, 600, 800" data="0.5, 0.72, 0.85, 0.91, 0.95" xLabel="Step" yLabel="Accuracy" >}}

**Exercise:** Implement DPO: derive the loss from the Bradley-Terry model and the optimal policy under a KL constraint. Train a language model directly on preference data without a separate reward model. Compare with PPO.

**Professor's hints**

- **DPO loss:** For each (prompt x, τ^w, τ^l), loss = -log σ(β * (log π(τ^w|x)/π_ref(τ^w|x) - log π(τ^l|x)/π_ref(τ^l|x))). Here β is a temperature (from the KL constraint). So we want the log-ratio for the preferred response to be higher than for the dispreferred. Implement log π(τ|x) as the sum of log probs of each token in τ given x and previous tokens.
- **Reference policy:** π_ref is the initial (or a fixed) LM; keep it frozen. Compute π_ref(τ|x) once per example and use in the loss.
- **Comparison with PPO:** Run both on the same preference data. PPO: train reward model (Bradley-Terry), then PPO with that reward + KL. DPO: train with DPO loss only. Compare (1) preference accuracy on held-out pairs, (2) reward model score on held-out prompts (if you have a reward model for eval), (3) training time and stability.
- Use a **small** LM (e.g. GPT-2) and short sequences so training is fast.

**Common pitfalls**

- **Numerical stability:** Log-ratios can be large; use log-sum-exp tricks or clamp. Ensure log π and log π_ref are computed in log space.
- **β (beta):** β controls how much we deviate from π_ref; larger β = stronger preference signal but more deviation. Tune (e.g. 0.1–0.5).
- **Tokenization:** Use the same tokenizer and context length for π and π_ref; τ^w and τ^l are token sequences.

{{< collapse summary="Worked solution (warm-up: DPO)" >}}
**Key idea:** DPO (Direct Preference Optimization) avoids training a separate reward model and then RL. We have (prompt, chosen response \\(\\tau^w\\), rejected response \\(\\tau^l\\)). We maximize the likelihood that \\(\\pi\\) assigns higher probability to \\(\\tau^w\\) than to \\(\\tau^l\\) (scaled by \\(\\pi/\\pi_{ref}\\) to stay close to reference). So we get a policy that reflects preferences without an explicit reward model or PPO loop. Simpler and often more stable than RLHF.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In DPO, why do we use the ratio π(τ)/π_ref(τ) instead of the reward r(τ)?
2. **Coding:** Implement DPO on 5k preference pairs (same as in Chapter 96). Train for 3 epochs. Evaluate: preference accuracy on 1k held-out pairs. Compare with PPO (reward model + 50 PPO steps): which achieves higher preference accuracy with less compute?
3. **Challenge:** Implement **IPO** (Identity Preference Optimization) or another DPO variant that changes the loss slightly (e.g. different normalization). Compare preference accuracy and generation quality with standard DPO.
