---
title: "Chapter 31: Introduction to Policy-Based Methods"
description: "When a stochastic policy is essential; why deterministic fails."
date: 2026-03-10T00:00:00Z
weight: 31
draft: false
---

**Learning objectives**

- Explain when a **stochastic policy** (outputting a distribution over actions) is essential versus when a deterministic policy suffices.
- Give a real-world scenario where a deterministic policy would fail (e.g. games with hidden information, adversarial settings).
- Relate stochastic policies to **exploration** and to **game AI** or **recommendation** where diversity matters.

**Concept and real-world RL**

**Policy-based methods** directly parameterize and optimize the policy \\(\pi(a|s;\theta)\\) instead of learning a value function and deriving actions from it. A **stochastic policy** outputs a probability over actions; a **deterministic policy** always picks the same action in a given state. In **game AI**, when the opponent can observe or anticipate your move (e.g. poker, rock-paper-scissors), a deterministic policy is exploitable—the opponent will always know what you do. A stochastic policy keeps the opponent uncertain and is essential for mixed strategies. In **recommendation**, showing a deterministic "best" item every time can create filter bubbles; stochastic policies (or sampling from a distribution) encourage exploration and diversity. For **robot navigation** in partially observable or noisy settings, randomness can help escape local minima or handle uncertainty.

**Where you see this in practice:** Stochastic policies are used in poker AI, multi-agent games, recommendation diversity, and any setting with hidden information or adversarial play. Deterministic policies (e.g. DDPG) are used when the environment is smooth and exploration is handled separately (e.g. noise).

**Exercise:** Discuss a scenario where a stochastic policy is essential (e.g., in games with hidden information). Explain why a deterministic policy would fail.

**Professor's hints**

- Think of a two-player game where the opponent can adapt: if your policy is deterministic, they can learn it and always counter. A stochastic policy makes you unpredictable.
- In recommendation, "always show the top-ranked item" can lead to no exploration of new items; sampling from a policy allows discovery.
- For the exercise, write 1–2 paragraphs: name the scenario, define state/action, and explain why determinism fails.

**Common pitfalls**

- **Assuming deterministic is always simpler:** In adversarial or partially observable settings, deterministic policies can be strictly worse; the "simplicity" of one action per state is a downside.
- **Confusing exploration with stochastic policy:** Exploration (e.g. ε-greedy) is one way to get randomness; a stochastic policy *is* a distribution over actions at every step, which naturally explores.

**Extra practice**

1. **Warm-up:** In one sentence each, give one example where a deterministic policy is fine and one where a stochastic policy is essential.
2. **Coding:** Write a Python function `stochastic_policy(probs)` that takes a list of action probabilities (summing to 1) and returns a sampled action index using `random.choices` or equivalent. Test with `[0.5, 0.3, 0.2]` over 1000 calls and approximate the empirical distribution.
3. **Challenge:** In rock-paper-scissors, what is the Nash-equilibrium policy? Is it deterministic or stochastic? Relate to the need for stochastic policies in games.
