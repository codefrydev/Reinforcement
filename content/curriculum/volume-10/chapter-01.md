---
title: "Chapter 91: RL in Robotics"
description: "Train in sim (e.g. arm reaching); domain randomization; sim-to-real."
date: 2026-03-10T00:00:00Z
weight: 91
draft: false
tags: ["robotics", "sim-to-real", "domain randomization", "curriculum"]
keywords: ["RL in robotics", "sim-to-real", "domain randomization", "arm reaching"]
---

**Learning objectives**

- **Train** a policy in **simulation** (e.g. robotic arm reaching or locomotion) using a standard RL algorithm (e.g. PPO or SAC).
- **Apply** **domain randomization**: vary physics parameters (e.g. mass, friction, motor gains) during training so the policy sees a distribution of sim environments.
- **Attempt** to **deploy** the policy in a real-world setting (or a different sim with "real" parameters) and **evaluate** the **sim-to-real gap** (drop in performance or need for adaptation).
- **Explain** why domain randomization can improve transfer: the policy becomes robust to parameter variation and may generalize to the real world.
- **Relate** sim-to-real and domain randomization to **robot navigation** and **healthcare** (safety-critical deployment).

**Concept and real-world RL**

**Reinforcement learning in robotics** often involves training in **simulation** (where we can run many episodes quickly) and then **deploying** in the real world. The **sim-to-real gap** is the performance drop or failure mode that occurs because the real world differs from the sim (dynamics, sensing, actuation). **Domain randomization** reduces this gap by training on many **randomized** sim instances (e.g. different masses, friction, delays); the policy learns to be robust to variation and may transfer to the real world. In **robot navigation** and **healthcare**, sim-to-real and safe deployment are central challenges.

**Where you see this in practice:** Domain randomization in robotics (OpenAI, etc.); sim-to-real for manipulation and locomotion; safety and verification before deployment.

**Illustration (sim-to-real gap):** With domain randomization, policies trained in sim can transfer to real. The chart below shows sim return vs real return (before and after randomization).

{{< chart type="bar" palette="comparison" title="Return: sim vs real (with domain rand)" labels="Sim, Real (no rand), Real (with rand)" data="95, 40, 85" yLabel="Return" >}}

**Exercise:** Train a policy in simulation (e.g., a robotic arm reaching) and then attempt to deploy it in a real-world setting by adding domain randomization (vary physics parameters). Evaluate the sim-to-real gap.

**Professor's hints**

- **Sim:** Use MuJoCo, PyBullet, or similar. Task: e.g. arm reaching (end-effector to target), or simple locomotion. Train with PPO or SAC until the policy succeeds in the default sim.
- **Domain randomization:** Each episode (or each reset), sample physics parameters from a range: e.g. link masses ±20%, friction ±30%, actuator gain ±20%. Train the same policy on these randomized envs. Optionally increase the range over training.
- **Deploy:** If you have real hardware, run the policy there and measure success rate or return. If not, create a "test" sim with fixed parameters different from training (e.g. different mass, friction) to simulate the "real" world and measure the sim-to-real gap (performance in default vs test params).
- **Evaluation:** Report mean return (or success rate) in (1) default sim, (2) randomized sim (average over 100 samples), (3) test / "real" params. Discuss the gap.

**Common pitfalls**

- **Over-randomization:** If parameters vary too much, the task may become impossible in some configs and the policy may not learn. Start with small ranges and increase gradually.
- **Under-randomization:** If the real world is outside the training distribution, transfer may still fail. Consider which parameters matter most (e.g. delay, friction) and randomize those.
- **No real robot:** The exercise can be completed with "test" sim parameters only; the key is to show that domain randomization improves robustness when evaluated on unseen parameters.

{{< collapse summary="Worked solution (warm-up: sim-to-real)" >}}
**Key idea:** We train in simulation with randomized parameters (physics, friction, masses, etc.). So the policy sees many "versions" of the world and learns to be robust. When we deploy in the real world (or in a test sim with different params), the policy generalizes better than if we had trained on a single fixed sim. Domain randomization is a simple but effective sim-to-real strategy; alternatives include system identification and domain adaptation.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might training with randomized physics parameters help when we deploy in the real world, where we do not know the exact parameters?
2. **Coding:** In MuJoCo (e.g. Reacher or FetchReach), train PPO with no randomization. Then train with domain randomization (mass, friction ±20%). Evaluate both policies on 50 episodes with a fixed "test" parameter set (e.g. +15% mass, -10% friction). Which policy generalizes better?
3. **Challenge:** Implement **system identification**: use a few real (or test-sim) rollouts to estimate physics parameters, then fine-tune the policy in sim with those parameters. Compare transfer with and without this step.
