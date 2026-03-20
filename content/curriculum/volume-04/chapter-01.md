---
title: "Chapter 31: Introduction to Policy-Based Methods"
description: "When a stochastic policy is essential; why deterministic fails."
date: 2026-03-10T00:00:00Z
weight: 31
draft: false
difficulty: 7
tags: ["policy gradient", "stochastic policy", "REINFORCE", "curriculum"]
keywords: ["policy-based methods", "stochastic policy", "deterministic policy", "exploration"]
roadmap_color: "amber"
roadmap_icon: "trend-up"
roadmap_phase_label: "Vol 4 · Ch 1"
---

**Learning objectives**

- Explain when a **stochastic policy** (outputting a distribution over actions) is essential versus when a deterministic policy suffices.
- Give a real-world scenario where a deterministic policy would fail (e.g. games with hidden information, adversarial settings).
- Relate stochastic policies to **exploration** and to **game AI** or **recommendation** where diversity matters.

**Concept and real-world RL**

**Policy-based methods** directly parameterize and optimize the policy \\(\pi(a|s;\theta)\\) instead of learning a value function and deriving actions from it. A **stochastic policy** outputs a probability over actions; a **deterministic policy** always picks the same action in a given state. In **game AI**, when the opponent can observe or anticipate your move (e.g. poker, rock-paper-scissors), a deterministic policy is exploitable—the opponent will always know what you do. A stochastic policy keeps the opponent uncertain and is essential for mixed strategies. In **recommendation**, showing a deterministic "best" item every time can create filter bubbles; stochastic policies (or sampling from a distribution) encourage exploration and diversity. For **robot navigation** in partially observable or noisy settings, randomness can help escape local minima or handle uncertainty.

**Where you see this in practice:** Stochastic policies are used in poker AI, multi-agent games, recommendation diversity, and any setting with hidden information or adversarial play. Deterministic policies (e.g. DDPG) are used when the environment is smooth and exploration is handled separately (e.g. noise).

**Illustration (policy distribution):** A stochastic policy outputs a probability over actions. The chart below shows an example distribution over 3 actions (e.g. left, stay, right) in one state.

{{< chart type="bar" title="π(a|s) for 3 actions" labels="Left, Stay, Right" data="0.2, 0.5, 0.3" >}}

**Exercise:** Discuss a scenario where a stochastic policy is essential (e.g., in games with hidden information). Explain why a deterministic policy would fail.

**Professor's hints**

- Think of a two-player game where the opponent can adapt: if your policy is deterministic, they can learn it and always counter. A stochastic policy makes you unpredictable.
- In recommendation, "always show the top-ranked item" can lead to no exploration of new items; sampling from a policy allows discovery.
- For the exercise, write 1–2 paragraphs: name the scenario, define state/action, and explain why determinism fails.

**Common pitfalls**

- **Assuming deterministic is always simpler:** In adversarial or partially observable settings, deterministic policies can be strictly worse; the "simplicity" of one action per state is a downside.
- **Confusing exploration with stochastic policy:** Exploration (e.g. ε-greedy) is one way to get randomness; a stochastic policy *is* a distribution over actions at every step, which naturally explores.

{{< collapse summary="Worked solution (warm-up: deterministic vs stochastic policy)" >}}
**Warm-up:** One example where deterministic is fine: a fully observed, deterministic environment (e.g. simple gridworld with no noise). One where stochastic is essential: when the optimal policy is mixed (e.g. rock-paper-scissors, or exploration in bandits). In RL we often use stochastic policies for exploration and for problems with multiple good actions; policy gradient methods learn a parameterized distribution \\(\\pi(a|s;\\theta)\\).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** In one sentence each, give one example where a deterministic policy is fine and one where a stochastic policy is essential.
2. **Coding:** Write a Python function `stochastic_policy(probs)` that takes a list of action probabilities (summing to 1) and returns a sampled action index using `random.choices` or equivalent. Test with `[0.5, 0.3, 0.2]` over 1000 calls and approximate the empirical distribution.
3. **Challenge:** In rock-paper-scissors, what is the Nash-equilibrium policy? Is it deterministic or stochastic? Relate to the need for stochastic policies in games.
4. **Variant:** Extend your stochastic policy function to work with 10 actions instead of 3. Verify empirically that for a uniform policy (`[0.1]*10`), all actions are visited roughly equally over 10000 calls.

{{< pyrepl code="import random\n\ndef stochastic_policy(probs):\n    return random.choices(range(len(probs)), weights=probs, k=1)[0]\n\n# Test with 3 actions\nprobs = [0.5, 0.3, 0.2]\ncounts = [0, 0, 0]\nfor _ in range(1000):\n    a = stochastic_policy(probs)\n    counts[a] += 1\nprint('Empirical:', [c/1000 for c in counts])\nprint('True:     ', probs)" height="200" >}}

5. **Debug:** The code below always picks action 0 because it uses `argmax` instead of sampling. Fix it to use proper stochastic selection.

```python
def broken_policy(probs):
    # BUG: argmax makes this deterministic
    return probs.index(max(probs))
```

6. **Conceptual:** In a fully observed, deterministic environment with no adversary, can the optimal policy always be deterministic? Explain why stochastic policies might still be useful for exploration.
7. **Recall:** State the definition of a stochastic policy \\(\\pi(a|s)\\) and a deterministic policy \\(\\mu(s)\\) from memory in one line each.
