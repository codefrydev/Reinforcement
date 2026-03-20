---
title: "Chapter 47: Soft Actor-Critic (SAC)"
description: "SAC for HalfCheetah with automatic temperature tuning."
date: 2026-03-10T00:00:00Z
weight: 47
draft: false
difficulty: 7
tags: ["SAC", "HalfCheetah", "temperature tuning", "curriculum"]
keywords: ["Soft Actor-Critic", "SAC", "HalfCheetah", "automatic temperature"]
roadmap_color: "purple"
roadmap_icon: "rocket"
roadmap_phase_label: "Vol 5 · Ch 7"
---

**Learning objectives**

- Implement **SAC** (Soft Actor-Critic) for **HalfCheetah**: two Q-networks (min for target), policy that maximizes \\(Q - \\alpha \\log \\pi\\), and **automatic temperature** tuning so \\(\\alpha\\) targets a desired entropy.
- **Train** and compare **sample efficiency** with PPO (same env, same or similar compute).

**Concept and real-world RL**

**SAC** combines maximum entropy RL with actor-critic: the critic learns two Q-functions (take min for target to reduce overestimation); the actor maximizes \\(\\mathbb{E}[ Q(s,a) - \\alpha \\log \\pi(a|s) ]\\); and \\(\\alpha\\) is updated to keep the policy entropy near a target (e.g. -\dim(a)). SAC is off-policy (replay buffer), so it is often more sample-efficient than PPO on continuous control. In **robot control** (HalfCheetah, Hopper, Walker), SAC is a standard baseline; in **recommendation** and **trading**, off-policy max-ent methods can improve exploration and stability.

**Where you see this in practice:** SAC is in OpenAI Spinning Up, Stable-Baselines3, and CleanRL; commonly used for MuJoCo benchmarks.

**Illustration (SAC learning curve):** On HalfCheetah, SAC typically reaches high return over 1M steps. The chart below shows mean return over last 10 episodes every 50k steps.

{{< chart type="line" palette="return" title="Mean return (SAC HalfCheetah)" labels="0, 250k, 500k, 750k, 1M" data="0, 1500, 3500, 4500, 5000" xLabel="Step" yLabel="Mean return" >}}

**Exercise:** Implement SAC for the HalfCheetah environment. Include automatic temperature tuning. Train and compare its sample efficiency with PPO (use existing implementations or your own).

**Professor's hints**

- Automatic \\(\\alpha\\): target entropy often \\(\\mathcal{H}_{target} = -\\dim(\\mathcal{A})\\). Loss for \\(\\alpha\\): \\(-\\alpha (\\log \\pi(a|s) + \\mathcal{H}_{target})\\). Minimize this so \\(\\alpha\\) increases when entropy is below target and decreases when above.
- Policy: Gaussian with squashing (tanh); log_prob must include the tanh Jacobian. Critic: input (s, a), output Q. Two critics, two target critics; target = \\(r + \\gamma (1-d) (\\min_i Q_i'(s', a') - \\alpha \\log \\pi(a'|s'))\\).
- HalfCheetah: state dim 17, action dim 6. Run for 1M steps (or your limit); plot mean return over last 10 episodes every 10k steps. Compare with PPO for same number of env steps.

**Common pitfalls**

- **Log_prob with squashing:** For \\(a = \\tanh(a_{raw})\\), \\(\\log \\pi(a) = \\log \\pi(a_{raw}) - \\sum \\log(1 - a^2)\\). Forgetting the correction gives wrong gradients.
- **Target entropy sign:** \\(\\mathcal{H}_{target}\\) is negative (e.g. -6 for dim 6). The \\(\\alpha\\) update uses \\(\\log \\pi + \\mathcal{H}_{target}\\); when entropy is low, \\(\\log \\pi\\) is large negative, so \\(\\log \\pi + \\mathcal{H}_{target}\\) is negative, and we want to increase \\(\\alpha\\) (so the loss for \\(\\alpha\\) is negative and we do gradient descent on \\(-\\alpha (\\ldots)\\) to increase \\(\\alpha\\)).

{{< collapse summary="Worked solution (warm-up: SAC alpha loss)" >}}
**Key idea:** SAC minimizes \\(J(\\alpha) = \\mathbb{E}[ -\\alpha (\\log \\pi(a|s) + \\mathcal{H}_{target}) ]\\). When entropy is below target, \\(\\log \\pi + \\mathcal{H}_{target} < 0\\), so \\(-\\alpha (\\ldots) > 0\\) for \\(\\alpha > 0\\); we increase \\(\\alpha\\) to weight entropy more and encourage exploration. When entropy is above target we decrease \\(\\alpha\\). So \\(\\alpha\\) adapts to keep entropy near the target.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** What is the role of the minimum of two Q-networks in SAC (same as in TD3)?
2. **Coding:** Implement SAC for HalfCheetah with auto \\(\\alpha\\). Log policy entropy and \\(\\alpha\\) every 1000 steps. Does entropy stay near the target?
3. **Challenge:** Run SAC and PPO each for 500k steps on HalfCheetah. Plot learning curves. Which reaches -300 (or better) return first in terms of env steps?
4. **Variant:** Run SAC with a fixed \\(\\alpha = 0.2\\) vs automatic temperature tuning on HalfCheetah. Does auto-tuning significantly improve final performance or convergence speed?
5. **Debug:** The SAC target computation below uses the wrong sign for the entropy term — it adds \\(\\alpha \\log \\pi\\) instead of subtracting it, punishing high-entropy actions. Fix it.

{{< pyrepl code="import torch\n\ndef sac_target_buggy(Q1_t, Q2_t, actor, s_next, r, done, alpha, gamma=0.99):\n    with torch.no_grad():\n        a_next, log_pi_next = actor.sample(s_next)\n        q_min = torch.min(Q1_t(s_next, a_next), Q2_t(s_next, a_next))\n        # BUG: should subtract alpha * log_pi, not add\n        target = r + gamma * (1 - done) * (q_min + alpha * log_pi_next)\n    return target\n\n# Fix: target = r + gamma * (1-done) * (q_min - alpha * log_pi_next)\nprint('Bug: + alpha*log_pi penalizes high-entropy actions (backwards)')\nprint('Fix: - alpha*log_pi encourages high entropy')" height="220" >}}

6. **Conceptual:** Why does SAC use two Q-networks? How does taking the minimum of two Q-estimates reduce overestimation compared to using a single critic?
7. **Recall:** Write the SAC actor loss \\(J_\\pi(\\theta) = \\mathbb{E}[\\ldots]\\) in terms of \\(Q, \\log \\pi, \\alpha\\) from memory.
