---
title: "Phase 8 Assessment: Advanced RL"
description: "15 questions spanning Volumes 6–10: model-based RL, exploration, offline RL, multi-agent, and real-world applications."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["assessment", "phase 8", "advanced RL", "self-check", "solutions"]
keywords: ["phase 8 assessment", "advanced RL quiz", "model-based MARL offline RLHF", "solutions"]
weight: 15
roadmap_icon: "brain"
roadmap_color: "violet"
roadmap_phase_label: "Phase 8 Quiz"
---

Use this self-check after completing Volumes 6–10 (Phase 8). These questions test conceptual understanding and the ability to connect ideas across topics.

---

### 1. Model-based vs model-free

Q: A robot has a perfect model of its environment. Should it use model-based or model-free RL, and why?

{{< collapse summary="Answer" >}}
**Model-based RL** — with a perfect model, the agent can plan (e.g. MCTS or value iteration) without needing real experience. This is far more sample-efficient. Model-free RL needs many real interactions to estimate value functions; model-based can compute them analytically or via simulation.
{{< /collapse >}}

---

### 2. Compounding error

Q: Why do multi-step model rollouts suffer from compounding error, and how does MBPO address this?

{{< collapse summary="Answer" >}}
Each model prediction introduces small errors. Over k steps, errors accumulate (compound). After many steps, the imagined trajectory diverges significantly from the true environment. **MBPO** uses short rollouts (typically 1–5 steps), keeping errors small while still generating useful synthetic data for the policy.
{{< /collapse >}}

---

### 3. MCTS phases

Q: Name and briefly describe the four phases of Monte Carlo Tree Search.

{{< collapse summary="Answer" >}}
1. **Selection**: traverse the tree from root to a leaf using UCB (or similar) to balance exploration/exploitation.
2. **Expansion**: add one or more child nodes to the selected leaf.
3. **Simulation** (rollout): from the new node, run a (random or heuristic) policy to a terminal state.
4. **Backpropagation**: update visit counts and value estimates along the path from the new node back to the root.
{{< /collapse >}}

---

### 4. Hard exploration

Q: Standard epsilon-greedy fails on Montezuma's Revenge (sparse reward). Why? Name one method designed for hard exploration.

{{< collapse summary="Answer" >}}
In Montezuma's Revenge, the agent must complete many specific actions before seeing any reward. Epsilon-greedy random exploration rarely stumbles upon this reward sequence. Methods: **ICM** (intrinsic curiosity — reward based on prediction error), **RND** (Random Network Distillation — intrinsic reward for novel states), **Go-Explore** (maintain and revisit frontier states).
{{< /collapse >}}

---

### 5. Offline RL distribution shift

Q: Why is distribution shift dangerous in offline RL? What does CQL do to address it?

{{< collapse summary="Answer" >}}
The offline dataset doesn't cover all state-action pairs. When the agent queries Q(s,a) for out-of-distribution (s,a), the function approximator can output arbitrarily large (overestimated) Q-values, leading to a policy that exploits these errors. **CQL** penalizes high Q-values for out-of-distribution actions by adding a regularization term to the loss: minimize Q for actions NOT in the dataset.
{{< /collapse >}}

---

### 6. Behavioral cloning vs DAgger

Q: What is the key difference between behavioral cloning and DAgger? When does behavioral cloning fail?

{{< collapse summary="Answer" >}}
**Behavioral cloning** trains on a fixed dataset of expert (state, action) pairs — supervised learning. It fails due to **distribution shift**: the learned policy makes small errors that put it in states not seen in training, where it makes larger errors, leading to compounding failures.

**DAgger** fixes this by iteratively running the current policy, querying the expert for correct actions in the newly visited states, and adding these to the training set. This reduces distribution shift by training on states the learner actually visits.
{{< /collapse >}}

---

### 7. Multi-agent credit assignment

Q: In CTDE (centralized training, decentralized execution), what does the critic have access to during training that the actors don't have during execution?

{{< collapse summary="Answer" >}}
During **training**, the centralised critic has access to the **observations and actions of all agents** (global state information). During **execution**, each actor only observes its own local observation. This allows the critic to better assign credit (knowing what all agents did), while keeping execution decentralised (each agent acts independently).
{{< /collapse >}}

---

### 8. RLHF pipeline

Q: Describe the three-step RLHF pipeline used to align LLMs.

{{< collapse summary="Answer" >}}
1. **Supervised fine-tuning (SFT)**: fine-tune the base LLM on high-quality demonstration data.
2. **Reward model training**: collect human preference comparisons (pairs of responses, human picks better one). Train a reward model to predict human preferences.
3. **RL fine-tuning**: use PPO (or similar) to optimize the LLM's policy against the reward model, with a KL penalty to prevent the policy from diverging too far from the SFT model.
{{< /collapse >}}

---

### 9. DPO vs PPO-RLHF

Q: How does DPO avoid training a separate reward model?

{{< collapse summary="Answer" >}}
DPO shows that the RLHF objective can be reparameterized in terms of the policy directly. The reward model is implicitly defined by the policy: r(x,y) = β log(π_θ(y|x)/π_ref(y|x)). DPO optimizes a cross-entropy loss on preference pairs using the policy ratio directly, eliminating the need to first train a reward model.
{{< /collapse >}}

---

### 10. Safe RL constraint

Q: In safe RL, what is a constraint violation? Give a concrete example in robotics.

{{< collapse summary="Answer" >}}
A constraint violation occurs when the agent takes an action that exceeds a safety limit, even if it increases reward. Example: a robot arm maximizing task performance might move too fast, exceeding joint velocity limits and damaging hardware. A safe RL algorithm would penalise or forbid actions with joint velocity > v_max, trading off some reward for safety.
{{< /collapse >}}

---

### 11. Sim-to-real gap

Q: What is the sim-to-real gap and name two techniques to reduce it?

{{< collapse summary="Answer" >}}
The sim-to-real gap is the discrepancy between a simulator's physics/dynamics and the real world. Policies trained in simulation may fail on real hardware due to this gap.

Techniques: **(1) Domain randomisation** — vary simulator parameters (friction, mass, noise) during training so the policy is robust to variation. **(2) System identification** — calibrate the simulator to match real hardware measurements.
{{< /collapse >}}

---

### 12. QMIX factorisation

Q: QMIX factorises the joint Q-function Q_tot as a monotone function of individual Q_i values. What constraint does monotonicity enforce, and why is it useful?

{{< collapse summary="Answer" >}}
Monotonicity: ∂Q_tot/∂Q_i ≥ 0 for all i. This means increasing any agent's individual Q-value can only increase the joint Q. This guarantees that **argmax over joint actions equals taking the argmax for each agent independently** — decentralised greedy execution is consistent with centralised optimal Q. Without monotonicity, greedy per-agent decisions might not achieve the global optimum.
{{< /collapse >}}

---

### 13. Entropy in SAC

Q: What does the entropy term in SAC's objective encourage? What happens when temperature α → 0?

{{< collapse summary="Answer" >}}
Entropy bonus H(π(·|s)) encourages the policy to remain **stochastic** (spread probability across multiple good actions), improving exploration and robustness. As α → 0, the entropy bonus vanishes and SAC converges to standard (non-maximum-entropy) RL — a deterministic greedy policy.
{{< /collapse >}}

---

### 14. Reward hacking

Q: Give one example of reward hacking in a real or hypothetical RL deployment.

{{< collapse summary="Answer" >}}
Many examples exist. One famous case: a boat-racing game agent found it could gain more reward by spinning in circles collecting power-ups than by completing the race. The reward (score from power-ups) was easier to maximize than the intended objective (finish the race). This illustrates how RL agents exploit gaps between the reward function and the designer's true intent.
{{< /collapse >}}

---

### 15. Meta-learning

Q: In MAML, what does "meta-learning" mean, and what is the inner loop vs outer loop?

{{< collapse summary="Answer" >}}
MAML learns an **initialisation θ** such that the model can quickly adapt to new tasks with a small number of gradient steps.

**Inner loop**: for each task, take k gradient steps from θ to get task-specific θ'. This is fast adaptation.

**Outer loop**: update θ so that the inner-loop adaptation achieves good performance across all tasks. The meta-objective is to find the best initialisation, not the best parameters for any one task.

**Score:** 12–15: Strong Phase 5 understanding. 9–11: Review specific volumes. Below 9: Return to the volumes covering missed topics before moving to research.
{{< /collapse >}}
