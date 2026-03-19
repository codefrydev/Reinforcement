---
title: "Chapter 100: The Future of Reinforcement Learning"
description: "Essay: foundation models and RL; architectures; path toward AGI."
date: 2026-03-10T00:00:00Z
weight: 100
draft: false
tags: ["future of RL", "foundation models", "AGI", "essay", "curriculum"]
keywords: ["future of reinforcement learning", "foundation models", "AGI", "RL architectures"]
---

**Learning objectives**

- **Write** a short **essay** (1–2 pages) on how **foundation models** (large pretrained models for language, vision, or multimodal) might **impact** reinforcement learning.
- **Discuss** potential **architectures** for decision-making that leverage large-scale pretraining (e.g. RL fine-tuning of LMs, world models with foundation model representations, or agents that use foundation models as policies or critics).
- **Speculate** on the **path toward AGI** (or toward more general and capable agents) from the perspective of RL + foundation models: what is missing, what might scale, and what risks or open problems remain.
- **Use** concepts from the curriculum (value functions, policy gradients, offline RL, multi-agent, safety, RLHF) where relevant.
- **Relate** to anchor scenarios (**robot navigation**, **game AI**, **recommendation**, **trading**, **healthcare**, **dialogue**) and where foundation models are already or could be applied.

**Concept and real-world RL**

**Foundation models** (LLMs, vision models, multimodal models) are pretrained on huge data and can be adapted to many tasks. Their impact on **RL** includes: (1) **Policies** or **value critics** that use pretrained representations or are themselves LMs (e.g. decision-making with LLMs, RLHF). (2) **World models** or **simulators** that use foundation models for state representation or dynamics. (3) **Scaling** RL with large models and large compute. The **path toward AGI** (or toward more general agents) may combine RL (trial-and-error, goals, long horizon) with foundation models (knowledge, language, generalization). In **dialogue**, **robot navigation**, and **game AI**, foundation models are already used with RL; the essay ties the curriculum to these trends.

**Where you see this in practice:** LLMs for decision-making; RLHF and alignment; world models and embodied AI; speculation and roadmaps (e.g. OpenAI, DeepMind blogs).

**Illustration (RL and foundation models):** Future RL may combine large-scale pretraining with decision-making. The chart below is a conceptual sketch: capability (e.g. return or task coverage) over model scale and RL fine-tuning.

{{< chart type="line" palette="return" title="Conceptual: capability vs scale + RL" labels="Pretrain, +RL 1M, +RL 10M, +RL 100M" data="0.2, 0.5, 0.75, 0.95" xLabel="Scale" yLabel="Capability" >}}

**Exercise:** Write a short essay (1–2 pages) on how foundation models might impact RL. Discuss potential architectures for decision-making that leverage large-scale pretraining, and speculate on the path toward AGI.

**Professor's hints**

- **Structure:** (1) Introduction: what are foundation models, what is RL; why combine them. (2) Impact: how might foundation models change RL (representations, policies, world models, scaling). (3) Architectures: 2–3 concrete ideas (e.g. LM as policy with RLHF; foundation model as feature extractor for value function; world model with LM-based state). (4) Path to AGI: what capabilities RL provides (goals, exploration, long-term planning) that pure pretraining may not; what is missing (e.g. safety, alignment, robustness). (5) Conclusion: summary and open questions.
- **Use curriculum concepts:** Refer to value functions, policy gradients, offline RL, multi-agent, safe RL, or RLHF where they fit. This shows integration of the course material.
- **Anchor scenarios:** Mention at least one or two (e.g. dialogue and RLHF; robot navigation and pretrained vision). Keeps the essay grounded.
- **Speculation:** The essay can be opinionated but should be reasoned; cite or allude to real work (e.g. GPT, AlphaGo, RLHF papers) where appropriate.

**Common pitfalls**

- **Too vague:** Avoid only high-level claims; include at least one or two concrete architecture or method ideas.
- **Ignoring RL:** The essay should be about RL + foundation models, not only about foundation models. RL brings goals, exploration, and sequential decision-making.
- **Overclaiming about AGI:** Speculation is fine, but avoid sounding like a press release. Acknowledge uncertainty and open problems.

{{< collapse summary="Worked solution (warm-up: future of RL)" >}}
**Key idea:** RL is used in games, robotics, recommendation, and LLM alignment. Open directions: more sample-efficient methods, better offline RL, safe and robust deployment, multi-agent and cooperative scaling, and combining RL with large models. The curriculum you completed (from bandits to deep RL, model-based, multi-agent, and applications) gives you the foundations to follow and contribute to these developments. Keep building and reading papers.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In 2–3 sentences, what can RL provide that pure supervised pretraining (e.g. next-token prediction) typically does not?
2. **Coding:** (Optional) Implement a minimal "LM as policy": use a small LM to output action logits for a discrete action space (e.g. map token ids to actions). Train with PPO on a simple env (e.g. CartPole). Does it learn? Discuss in one paragraph how this scales to larger LMs and harder tasks.
3. **Challenge:** Write a second essay section (1 page) on **risks** of combining foundation models and RL (e.g. misuse, alignment, scaling without safety). Compare with risks of RL alone and of foundation models alone.
4. **Variant:** LLM-as-agent can be implemented in two ways: (a) the LLM directly outputs actions via token generation, or (b) the LLM produces a chain-of-thought plan that is then executed. Describe the trade-offs in sample efficiency, interpretability, and action space expressiveness between these two designs.
5. **Debug:** An LLM agent that uses tool calls (e.g. a calculator or search API) achieves high reward during training but fails at evaluation when a different version of the tool is deployed. The reward function during training evaluated tool outputs from the training-time tool, not the actual task outcome. Explain this form of reward hacking and how to design reward functions that are robust to changes in the tool API.
6. **Conceptual:** Foundation models trained with RL (e.g. via RLHF or GRPO) can exhibit emergent capabilities at scale that were not present at smaller sizes. Discuss why this makes safety evaluation challenging: if capabilities emerge unpredictably at scale, what does this imply for testing and deploying RL-trained foundation models responsibly?
