---
title: "Chapter 95: Training Large Language Models with PPO"
description: "PPO fine-tune small LM (e.g. GPT-2) for sentiment; KL penalty."
date: 2026-03-10T00:00:00Z
weight: 95
draft: false
tags: ["PPO", "LLM", "GPT-2", "sentiment", "KL penalty", "curriculum"]
keywords: ["PPO for LLM", "training language models", "GPT-2", "sentiment", "KL penalty"]
---

**Learning objectives**

- **Implement** a **PPO loop** to fine-tune a **small language model** (e.g. GPT-2 small or DistilGPT-2) for text generation with a **simple reward** (e.g. positive sentiment, or length).
- **Include** a **KL penalty** (or KL constraint) so that the updated policy does not deviate too far from the initial (reference) policy, preventing **mode collapse** and maintaining fluency.
- **Generate** sequences with the current policy, compute reward for each sequence, and update the policy with PPO (clip + KL).
- **Observe** that without KL penalty the policy may collapse (e.g. always output the same high-reward token); with KL it stays diverse.
- **Relate** to **dialogue** and **RLHF**: same PPO+KL setup is used for aligning LMs with human preferences.

**Concept and real-world RL**

**Training LMs with PPO** uses reinforcement learning to optimize a **reward** over generated text (e.g. sentiment score, task success, or learned reward from human preferences). The **policy** is the LM (autoregressive); we sample sequences, score them with the reward, and update the LM with **PPO** to maximize expected reward. A **KL penalty** (or constraint) limits how much the policy can change from a **reference** LM, which prevents **mode collapse** (e.g. always outputting "positive positive positive") and keeps generations fluent. In **dialogue** and **RLHF**, this is the core of aligning large language models with human intent.

**Where you see this in practice:** PPO for LM fine-tuning (InstructGPT, ChatGPT); KL-constrained RLHF; sentiment and style control.

**Illustration (PPO for LM):** When fine-tuning an LM with PPO, reward (e.g. sentiment) and KL to the reference model are tracked. The chart below shows reward and KL over PPO steps.

{{< chart type="line" palette="return" title="Reward and KL (PPO-LM)" labels="0, 100, 200, 300, 400" data="0.5, 0.7, 0.82, 0.88, 0.92" xLabel="Step" yLabel="Reward" >}}

**Exercise:** Using a small language model (e.g., GPT-2), implement a PPO loop to fine-tune it for text generation with a simple reward (e.g., positive sentiment). Include a KL penalty to prevent mode collapse.

**Professor's hints**

- **LM:** Use Hugging Face (e.g. gpt2 or distilgpt2). The policy π(a|s) is the LM: s = prompt + generated so far, a = next token. Sample a batch of completions (e.g. 64 sequences, max length 32).
- **Reward:** Use a sentiment classifier (e.g. from Hugging Face) to score each sequence; reward = probability of "positive" or score in [-1,1]. Or reward = length (to encourage longer generations) or a simple heuristic.
- **PPO:** Compute log π(a|s) for the current policy and for the reference (initial) policy. Advantage = reward - baseline (e.g. mean reward, or a value model). PPO loss = clip(ratio, 1-ε, 1+ε) * advantage; ratio = π/π_ref. Add KL term: β * KL(π || π_ref) so we penalize deviation from π_ref.
- **KL penalty:** KL ≈ (log π - log π_ref). Average over tokens in the batch. Tune β: too large and the policy barely moves; too small and mode collapse. Start with β=0.01–0.1.
- **Mode collapse:** Without KL, the policy may output the same short high-reward sequence every time. With KL, it should stay close to the reference and remain diverse.

**Common pitfalls**

- **Reward scale:** Normalize rewards (e.g. subtract mean, divide by std) so advantages are stable.
- **Baseline:** Use a value model (train V(s) to predict reward) or simple mean reward as baseline to reduce variance.
- **Sequence-level reward:** The reward is usually for the whole sequence; for PPO we need per-token or per-step credit. Use the same reward for all steps in the sequence (reward-to-go from that step), or use a value model for baseline.

{{< collapse summary="Worked solution (warm-up: RL for LLMs)" >}}
**Key idea:** We train an LLM policy to generate text; reward is given at the end of the sequence (e.g. quality, safety). We use PPO: sample a completion, get reward, compute per-token log-probs and advantages (e.g. reward-to-go or a value model as baseline). The policy gradient update increases the probability of tokens in high-reward sequences. We keep a reference policy and use a KL penalty so the policy does not drift too far (avoid mode collapse and reward hacking).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why does a KL penalty between the current policy and the reference policy help prevent mode collapse?
2. **Coding:** Fine-tune GPT-2 with PPO for 100 iterations. Reward = sentiment score (positive = 1, negative = -1). Plot mean reward and mean KL(π || π_ref) per iteration. Compare with no KL (β=0): does the policy collapse to a few tokens?
3. **Challenge:** Use a **value model** V(prompt) trained to predict expected reward. Use it as the baseline in PPO. Does it speed up learning or reduce variance compared to using mean reward as baseline?
4. **Variant:** Vary the KL penalty coefficient β ∈ {0, 0.01, 0.1, 1.0}. Plot both reward and generation diversity (e.g. distinct n-gram ratio) for each β. What value balances reward maximization and linguistic diversity?
5. **Debug:** RLHF with PPO achieves high sentiment reward but the model starts generating grammatically broken text. Logging shows KL divergence from the reference model has grown to 50 nats. The KL penalty was applied per token but the coefficient β was not scaled by sequence length, making it effectively 50× smaller than intended. Describe the correct per-token vs per-sequence KL scaling.
6. **Conceptual:** In RLHF for LLMs, the KL penalty β serves two purposes: preventing reward hacking and preserving language model fluency. Explain why simply training with a high reward signal and no KL penalty leads to degenerate outputs even when the reward function is accurate, using the concept of mode-seeking behavior in the policy optimization.
