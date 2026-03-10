---
title: "Chapter 11: Monte Carlo Methods"
description: "First-visit MC prediction for blackjack."
date: 2026-03-10T00:00:00Z
weight: 11
draft: false
tags: ["Monte Carlo", "prediction", "blackjack", "curriculum"]
keywords: ["Monte Carlo methods", "first-visit MC", "blackjack", "prediction"]
---

**Learning objectives**

- Implement first-visit Monte Carlo prediction: estimate \\(V^\\pi(s)\\) by averaging returns from the first time \\(s\\) is visited in each episode.
- Use a Gym/Gymnasium blackjack environment and a fixed policy (stick on 20/21, else hit).
- Interpret value estimates for key states (e.g. usable ace, dealer showing 10).

**Concept and real-world RL**

**Monte Carlo (MC) methods** estimate value functions from experience: run episodes under a policy, compute the return from each state (or state-action), and average those returns. **First-visit MC** uses only the first time each state appears in an episode; **every-visit MC** uses every visit. No model (transition probabilities) is needed—only sample trajectories. In RL, MC is used when we can get full episodes (e.g. games, episodic tasks) and want simple, unbiased estimates. **Game AI** is a natural fit: blackjack has a small state space (player sum, dealer card, usable ace), stochastic transitions (card draws), and a clear "stick or hit" policy to evaluate. The same idea applies to evaluating a fixed strategy in any episodic game—we run many episodes and average the returns from each state.

**Where you see this in practice:** MC prediction is used to evaluate policies in card games, board games, and simulators where we can run full episodes. It is a building block for MC control and appears in benchmarking and policy evaluation in industry.

**Exercise:** Implement first-visit Monte Carlo prediction for the blackjack environment (OpenAI Gym). Estimate the state-value function for a policy that sticks on 20 or 21, otherwise hits. Run for 500,000 episodes and plot the value for a few key states.

**Professor's hints**

- Blackjack state is often (player_sum, dealer_card, usable_ace). Use a dict or array keyed by state to store (sum of returns, count) for first-visit averages. When a state is first seen in an episode, append the *return from that step* to its list (or add to sum and increment count).
- Return from step \\(t\\): \\(G_t = r_{t+1} + \\gamma r_{t+2} + \\cdots\\) until the end of the episode. Compute this by looping backward from the end of the episode (or store rewards and discount forward).
- Policy: if player sum is 20 or 21, action 0 (stick); else action 1 (hit). Run many episodes; after 500k, \\(V(s)\\) = total return from first visits to \\(s\\) / count of first visits.
- Plot: e.g. \\(V\\) for (player_sum=20, dealer_card=10, usable_ace=False) and a few other (sum, dealer, ace) combinations. You can use a heatmap or bar chart for a subset of states.

**Common pitfalls**

- **First-visit vs every-visit:** First-visit uses at most one return per state per episode. Every-visit uses every time the state appears; estimates differ slightly. The exercise asks for first-visit.
- **Return from first occurrence:** The return \\(G_t\\) to use is the return *from time \\(t\\) onward*, not the return from the start of the episode. So you need to compute partial returns from each first-visit index to the end.
- **State representation:** Gym's Blackjack may return a tuple (sum, dealer, ace). Use it as the key for your \\(V\\) table; do not mix up the order (e.g. (dealer, sum) would be wrong).

{{< collapse summary="Worked solution (warm-up: one episode, first-visit returns)" >}}
**Warm-up:** For one episode with the given policy, list the states visited and the return from the first time each state is visited. Example for a 3-step episode: suppose states are (18, 6, False), (19, 6, False), (20, 6, False) and rewards are 0, 0, 1 (win at end). **Step 1:** Returns from first visit: from (18,6,F) use \\(G_0 = r_1 + \\gamma r_2 + \\gamma^2 r_3 = 0 + 0.9\\cdot 0 + 0.81\\cdot 1 = 0.81\\); from (19,6,F) use \\(G_1 = r_2 + \\gamma r_3 = 0.9\\); from (20,6,F) use \\(G_2 = r_3 = 1\\). **Step 2:** So first-visit returns are (18,6,F)→0.81, (19,6,F)→0.9, (20,6,F)→1. Each state gets one return per episode in first-visit MC; we average these over many episodes to estimate \\(V^\\pi\\).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For one episode of blackjack with the given policy, list the states visited and the return from the first time each state is visited. Compute returns by hand for a 3-step episode.
2. **Coding:** Write a function that, given a list of (state, reward) pairs for one episode and gamma, returns the partial return G_t from each first-visit state. Test on a 5-step episode.
3. **Challenge:** Implement **every-visit** MC for the same policy. Compare first-visit and every-visit estimates for 2–3 states after 100k episodes. Are they similar?
