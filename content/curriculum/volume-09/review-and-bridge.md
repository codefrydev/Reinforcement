---
title: "Volume 9 Review & Bridge to Volume 10"
description: "Review Volume 9 (Multi-Agent RL, game theory, QMIX, MAPPO) and preview Volume 10 (Real-World RL — safety, alignment, LLMs, deployment)."
date: 2026-03-19T00:00:00Z
draft: false
difficulty: 8
weight: 100
tags: ["review", "bridge", "Volume 9", "Volume 10", "multi-agent", "QMIX", "MAPPO", "safety", "RLHF", "LLMs"]
roadmap_color: "blue"
roadmap_icon: "network"
roadmap_phase_label: "Vol 9 · Review"
---

## Volume 9 Recap Quiz (5 questions)

{{< collapse summary="Q1. What is a Nash Equilibrium, and why is it hard to find in multi-agent RL?" >}}
A Nash Equilibrium is a joint policy (π₁*, π₂*, ..., πₙ*) where no single agent can improve its reward by unilaterally changing its policy (given the others are fixed). It's hard to find in MARL because: (1) each agent's environment is non-stationary (other agents are learning simultaneously); (2) there may be multiple Nash equilibria; (3) gradient-based methods may cycle or diverge in competitive settings; (4) the joint action space is exponential in the number of agents.
{{< /collapse >}}

{{< collapse summary="Q2. What is IQL (Independent Q-Learning) and why does it often work despite being theoretically flawed?" >}}
IQL treats each agent as an independent learner — each has its own Q-network and ignores other agents' actions/policies. Theoretically, the environment is non-stationary from each agent's perspective, so convergence guarantees break. In practice it often works because: (1) the other agents' policies change slowly; (2) the team reward provides enough signal; (3) it's simple and parallelisable. It fails when tight coordination is needed.
{{< /collapse >}}

{{< collapse summary="Q3. What is CTDE (Centralised Training, Decentralised Execution)?" >}}
During **training**: agents have access to global state, other agents' actions/observations, and can share gradients — this enables coordination. During **execution**: each agent acts using only its own local observations (no communication or global state needed). This is practical for real-world deployment where agents are distributed. QMIX, MADDPG, and MAPPO all follow CTDE.
{{< /collapse >}}

{{< collapse summary="Q4. How does QMIX enforce monotonic mixing without losing expressiveness?" >}}
QMIX factorises the joint Q-function as Q_tot = f(Q₁(τ₁,a₁), ..., Qₙ(τₙ,aₙ)) where f is a **monotonically increasing** function (with non-negative weights from a hypernetwork). This guarantees that argmax Q_tot = (argmax Q₁, ..., argmax Qₙ) — decentralised greedy is optimal. QMIX doesn't need joint action enumeration, making it scalable. The monotonicity constraint is limiting for non-cooperative settings.
{{< /collapse >}}

{{< collapse summary="Q5. What is the difference between cooperative, competitive, and mixed multi-agent settings?" >}}
- **Cooperative**: all agents share the same reward. Goal: maximise joint return. Algorithms: QMIX, MAPPO, QPLEX.
- **Competitive** (zero-sum): one agent's gain is another's loss. Game tree search, self-play (AlphaZero), Nash Q-learning.
- **Mixed** (general-sum): agents have different reward functions, partially aligned. Most real-world settings. Requires general game-theoretic approaches; no single dominant paradigm.
{{< /collapse >}}

---

## What Changes in Volume 10

| | Volume 9 (Academic / Game Settings) | Volume 10 (Real-World Deployment) |
|---|---|---|
| **Environment** | Simulated game / benchmark | Physical world, production systems, LLMs |
| **Safety** | Not a concern (reset easily) | Critical — unsafe actions have real consequences |
| **Reward** | Well-defined (game score) | Ambiguous — requires human feedback or IRL |
| **Distribution** | Stationary at test time | Distributional shift, adversarial inputs |
| **Interpretability** | Optional | Often legally / ethically required |
| **Scale** | Hundreds of agents | Billions of parameters (LLM fine-tuning) |

**The big insight:** RL in the real world requires safety constraints, interpretable policies, robust behaviour under distribution shift, and alignment with human preferences. RLHF (Reinforcement Learning from Human Feedback) is the dominant method for aligning large language models — it is PPO applied to token generation with a learned human preference reward model.

---

## Bridge Exercise: Reward Hacking — The Alignment Problem in Miniature

{{< pyrepl code="# Reward hacking: when an agent finds a way to maximise\n# the proxy reward without achieving the true goal.\n\nimport random\nrandom.seed(7)\n\nclass TidyingRobot:\n    \"\"\"\n    True goal: have a clean room.\n    Proxy reward: number of visible dirty items = 0.\n    Unintended solution: cover everything with a blanket!\n    \"\"\"\n    def __init__(self):\n        self.dirty_items = 10\n        self.items_hidden = 0\n\n    def clean_item(self):\n        \"\"\"Properly cleans one item. Slow but correct.\"\"\"\n        if self.dirty_items > 0:\n            self.dirty_items -= 1\n        return 1  # reward: one fewer visible dirty item\n\n    def cover_with_blanket(self):\n        \"\"\"Hides all items. Fast but cheating!\"\"\"\n        hidden = self.dirty_items\n        self.items_hidden += hidden\n        self.dirty_items = 0\n        return hidden  # reward: ALL items \"cleaned\" at once\n\n    def proxy_reward(self):\n        return -self.dirty_items  # proxy: minimize visible dirty items\n\n    def true_reward(self):\n        return -(self.dirty_items + self.items_hidden)  # true: minimize ALL dirty items\n\nrobot = TidyingRobot()\nprint(f'Initial: dirty={robot.dirty_items}, proxy={robot.proxy_reward()}, true={robot.true_reward()}')\n\n# Reward-hacking agent chooses cover_with_blanket (maximises proxy in 1 step)\nrobot.cover_with_blanket()\nprint(f'After blanket: dirty={robot.dirty_items}, hidden={robot.items_hidden}')\nprint(f'  Proxy reward: {robot.proxy_reward()}  (perfect! 0 visible dirty items)')\nprint(f'  True reward:  {robot.true_reward()}  (terrible! still 10 dirty items)')\nprint()\nprint('Volume 10 tackles this: RLHF, reward modeling, constrained RL, interpretability.')" height="340" >}}

**Next:** [Volume 10: Real-World RL, Safety & LLM Alignment](../volume-10/)
