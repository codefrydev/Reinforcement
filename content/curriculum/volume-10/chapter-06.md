---
title: "Chapter 96: Implementing RLHF in NLP"
description: "Simulated preference data; Bradley-Terry reward model; PPO finetune."
date: 2026-03-10T00:00:00Z
weight: 96
draft: false
tags: ["RLHF", "preference data", "Bradley-Terry", "PPO", "NLP", "curriculum"]
keywords: ["RLHF implementation", "preference data", "Bradley-Terry", "reward model", "PPO"]
---

**Learning objectives**

- **Collect** (or simulate) **human preference data**: pairs of model responses to the same prompt, with a label indicating which response is preferred.
- **Train** a **reward model** using the **Bradley-Terry** loss: P(τ^w preferred over τ^l) = σ(r(τ^w) - r(τ^l)), where r is the reward model (e.g. LM that outputs a scalar or a separate head).
- **Fine-tune** the **language model** with **PPO** using the learned reward model as the reward (and a KL penalty to the initial LM).
- **Evaluate** on held-out prompts: generate with the fine-tuned LM and score with the reward model; optionally compare with the initial LM.
- **Relate** to the **dialogue** anchor and real RLHF pipelines (InstructGPT, etc.).

**Concept and real-world RL**

**RLHF in NLP** has three steps: (1) **Collect preferences**: humans (or a proxy) compare two responses to the same prompt and say which is better. (2) **Train a reward model**: fit r so that P(τ^w preferred) = σ(r(τ^w) - r(τ^l)) (Bradley-Terry). (3) **Fine-tune the LM with RL**: use PPO to maximize expected r(τ) when generating τ, with a KL penalty to the initial LM so it stays on-distribution. In **dialogue**, this aligns the LM with human preferences (helpful, harmless, etc.) without hand-writing a reward function. This chapter implements the full pipeline with simulated preferences.

**Where you see this in practice:** InstructGPT, ChatGPT, Claude; Bradley-Terry reward models; PPO for LM alignment.

**Exercise:** Collect human preference data (simulated) for two responses from a language model. Train a reward model using the Bradley-Terry loss. Then fine-tune the LM with PPO using that reward model.

**Professor's hints**

- **Simulated preferences:** Generate pairs (prompt, response_A, response_B) with the current LM (or a fixed LM). Label: prefer A if reward_true(A) > reward_true(B), where reward_true is a simple proxy (e.g. length, sentiment, or "contains keyword"). So you do not need human labelers; the proxy is the "human."
- **Reward model:** r_ψ(prompt, response) → scalar. Can be an LM that takes "[prompt] [response]" and has a scalar head, or a separate classifier. Train with loss = -log σ(r(τ^w) - r(τ^l)) on the preference dataset.
- **PPO fine-tune:** Same as Chapter 95: sample responses from current π, score with r_ψ, update π with PPO + KL to π_ref. The reward is r_ψ(prompt, response). Run for several iterations; the reward model can be frozen or periodically updated (in full RLHF it is usually frozen during PPO).
- **Evaluation:** On held-out prompts, generate with π and compute mean r_ψ. Compare with π_ref. Also check that KL(π || π_ref) does not explode (generations stay coherent).

**Common pitfalls**

- **Reward hacking:** The LM may exploit the reward model (e.g. repeat tokens that get high r). KL penalty and a good reward model (trained on diverse preferences) help.
- **Overfitting the reward model:** If the reward model overfits the preference set, it may not generalize; use a held-out set and early stopping.
- **Data size:** Simulated preferences can be generated in bulk; use at least a few thousand pairs for the reward model.

{{< collapse summary="Worked solution (warm-up: reward model from preferences)" >}}
**Key idea:** We have preference data: (response A, response B, preferred). We train a reward model \\(r_\\psi\\) so that \\(P(A > B) = \\sigma(r(A) - r(B))\\) (Bradley-Terry). Loss = cross-entropy between predicted preference and actual. We need enough pairs (e.g. thousands) so the reward model generalizes. Then we use \\(r_\\psi\\) in RLHF to optimize the policy. The reward model is a proxy for human preference and can have biases from the data.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why do we train a reward model from preferences instead of using the preference labels directly as the reward in PPO?
2. **Coding:** Generate 5k (prompt, response_A, response_B) with a fixed LM; label by sentiment (prefer more positive). Train reward model (Bradley-Terry). Fine-tune the LM with PPO for 50 iterations. Plot mean reward (from reward model) on eval prompts and mean KL. Does the LM improve on the reward model score?
3. **Challenge:** Use **best-of-N** or **rejection sampling** as a baseline: generate N responses per prompt and pick the one with highest r. Compare with PPO: which gives higher reward on eval? Which is more sample-efficient (number of LM forward passes)?
