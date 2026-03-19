---
title: "Checkpoint: Volume 2, Midpoint (After Chapter 15)"
description: "5 quick questions after Chapters 11–15 of Volume 2. Check you're ready to continue."
date: 2026-03-19T00:00:00Z
draft: false
tags: ["checkpoint", "volume 2", "assessment", "Monte Carlo", "TD", "SARSA"]
---

Take this checkpoint after completing Chapters 11–15 (Monte Carlo methods, TD learning, SARSA). All 5 should feel manageable — if any are unclear, re-read the relevant chapter before continuing.

---

**Q1.** What is the Monte Carlo estimate of V(s)?

{{< collapse summary="Answer" >}}
The Monte Carlo estimate of V(s) is the **average of all observed returns** from visits to state s:

V(s) ← average of G_t for all t where S_t = s.

More precisely, after each episode we observe a return G_t for each visit to s, and V(s) is updated toward the mean of those returns. MC waits until the episode ends to compute G_t — it uses no bootstrapping.
{{< /collapse >}}

---

**Q2.** Write the TD(0) update rule from memory.

{{< collapse summary="Answer" >}}
V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) − V(S_t)]

Where:
- **α** is the learning rate (step size),
- **R_{t+1} + γ V(S_{t+1})** is the TD target,
- **R_{t+1} + γ V(S_{t+1}) − V(S_t)** is the TD error (δ).

TD(0) updates after every step using the next state's current value estimate — no need to wait for the episode to end.
{{< /collapse >}}

---

**Q3.** Apply one TD(0) update by hand: V(A) = 0.4, reward r = 0, V(B) = 0.6, α = 0.1, γ = 0.9. What is the new V(A)?

{{< collapse summary="Answer" >}}
TD target = r + γ V(B) = 0 + 0.9 × 0.6 = **0.54**

TD error = 0.54 − 0.4 = **0.14**

New V(A) = 0.4 + 0.1 × 0.14 = **0.414**
{{< /collapse >}}

---

**Q4.** What is the key difference between TD and Monte Carlo in terms of bootstrapping?

{{< collapse summary="Answer" >}}
- **Monte Carlo** does **not bootstrap**: it waits until the end of the episode and uses the actual full return G_t to update V(s). No estimates are used inside the update.
- **TD** does **bootstrap**: it updates V(s) using the estimated value of the next state V(S_{t+1}), which is itself an approximation. TD learns from incomplete episodes and updates at every step.

Bootstrapping = using your own current estimates as targets. MC uses real data; TD uses estimated data.
{{< /collapse >}}

---

**Q5.** Why does SARSA tend to take safer paths than Q-learning in environments with cliffs or traps?

{{< collapse summary="Answer" >}}
- **SARSA** is **on-policy**: it updates Q(s, a) using the action actually taken next (including ε-greedy exploratory steps). If the policy occasionally explores toward a cliff, those dangerous transitions are backed up into Q-values, making risky states look less attractive.
- **Q-learning** is **off-policy**: it updates using max_a Q(s', a) — the greedy best action — regardless of what was actually taken. It learns the optimal greedy policy, which may walk close to the cliff because exploration accidents are not factored into the update.

Result: Under an ε-greedy policy, SARSA learns a **safer** path that avoids danger even when exploring; Q-learning learns the shortest path but may fall off cliffs during training.
{{< /collapse >}}

---

All 5 correct? Continue to Chapter 16 (Q-learning and off-policy TD). Stuck on 2 or more? Re-read Chapters 12–14.
