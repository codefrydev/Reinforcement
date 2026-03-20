---
title: "Volume 3 Review & Bridge to Volume 4"
description: "Review Volume 3 (DQN and variants) and preview Volume 4 (Policy Gradients). From value-based to policy-based methods."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 7
weight: 100
tags: ["review", "bridge", "Volume 3", "Volume 4", "DQN", "policy gradient"]
roadmap_color: "green"
roadmap_icon: "chart"
roadmap_phase_label: "Vol 3 · Review"
---

## Volume 3 Recap Quiz (5 questions)

{{< collapse summary="Q1. What two techniques does DQN add to basic Q-learning to stabilize training?" >}}
1. **Experience replay**: store transitions in a buffer, sample random mini-batches (breaks temporal correlation).
2. **Target network**: a separate copy of Q, updated less frequently — prevents the target from shifting every step.
{{< /collapse >}}

{{< collapse summary="Q2. What is the 'deadly triad'?" >}}
The combination of **function approximation + bootstrapping + off-policy learning**. Each alone is fine; together they can cause divergence in Q-learning with neural networks.
{{< /collapse >}}

{{< collapse summary="Q3. What does Dueling DQN decompose, and why?" >}}
Q(s,a) = V(s) + A(s,a). V(s) captures the value of the state regardless of action; A(s,a) captures the relative value of each action. This helps when many actions have similar Q-values — the network can learn V accurately even when action effects are small.
{{< /collapse >}}

{{< collapse summary="Q4. Why does Double DQN reduce overestimation?" >}}
Standard DQN uses the same network to select and evaluate the action: max_{a'} Q(s',a'; θ⁻). This overestimates Q because the max of noisy estimates is biased upward. Double DQN separates these: use the online network to SELECT the action (argmax), use the target network to EVALUATE it. This decorrelates selection and evaluation.
{{< /collapse >}}

{{< collapse summary="Q5. What is the main limitation of DQN for continuous action spaces?" >}}
DQN requires computing max_a Q(s,a) over all actions (at every step). For continuous actions (e.g. joint torques), this max is intractable — infinite actions. Volume 4 introduces policy gradient methods that can directly output continuous actions.
{{< /collapse >}}

---

## What Changes in Volume 4

| | Volume 3 (Value-based) | Volume 4 (Policy-based) |
|---|---|---|
| **What is parameterized** | Q(s,a; θ) | π(a\|s; θ) directly |
| **Action space** | Discrete (DQN takes argmax) | Discrete or continuous |
| **Update signal** | TD error (supervised-like) | Policy gradient (REINFORCE) |
| **On/Off-policy** | Off-policy (replay) | Mostly on-policy |
| **Sample efficiency** | Better (replay buffer) | Worse (needs fresh data) |

**The big insight:** Instead of learning Q and deriving the policy, directly parameterise π and optimise it. This enables continuous control (robotics) and naturally stochastic policies (games with hidden info).

---

## Bridge Exercise: Why DQN Fails on Continuous Actions

{{< pyrepl code="import numpy as np\n\n# DQN on discrete: max over 4 actions (fast)\nQ_discrete = np.array([0.3, 0.7, -0.1, 0.5])\nbest_action = np.argmax(Q_discrete)\nprint(f'Discrete argmax: action {best_action} (Q={Q_discrete[best_action]})')\n\n# DQN on continuous: we'd need to maximise Q(s, a) over a ∈ [-1, 1]\n# This requires optimisation at EVERY step -- impractical\n# Policy gradient instead: directly output action mean + std\nimport random\n\ndef gaussian_policy(mean, std):\n    \"\"\"Sample action from Normal(mean, std).\"\"\"\n    return random.gauss(mean, std)\n\nmean, std = 0.3, 0.1   # learned parameters\nactions = [gaussian_policy(mean, std) for _ in range(5)]\nprint(f'Continuous policy samples: {[round(a,3) for a in actions]}')\nprint('No argmax needed -- just sample from the distribution!')" height="280" >}}

**Next:** [Volume 4: Policy Gradients](../volume-04/)
