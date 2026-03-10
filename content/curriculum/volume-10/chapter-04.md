---
title: "Chapter 94: RL in Recommender Systems"
description: "Toy recommender, 100 items, changing user; maximize engagement."
date: 2026-03-10T00:00:00Z
weight: 94
draft: false
tags: ["recommender systems", "engagement", "RL", "curriculum"]
keywords: ["RL in recommender systems", "recommendation", "engagement", "changing user"]
---

**Learning objectives**

- **Build** a **toy recommender**: 100 items, a **user model** with **changing preferences** (e.g. latent state that drifts or has context-dependent taste).
- **Define** state (e.g. user history, current context), action (which item to show), and reward (e.g. click, watch time, or engagement score).
- **Train** an agent with a **policy gradient** method (e.g. REINFORCE or PPO) to maximize **long-term engagement** (e.g. cumulative clicks or cumulative reward over a session).
- **Compare** with a baseline (e.g. random or greedy to current preference) and report engagement over episodes.
- **Relate** the formulation to the **recommendation** anchor (state = user context, action = item, return = long-term satisfaction).

**Concept and real-world RL**

**RL in recommender systems** treats recommendation as a sequential decision problem: at each step we observe **state** (user history, context) and choose an **action** (which item to show); we get a **reward** (click, watch time, purchase) and the user state may change. The goal is to maximize **long-term engagement** (discounted sum of rewards), not just immediate reward. A **toy** setting might have 100 items, a user with a latent preference that can drift or depend on context, and a simple reward (e.g. 1 if click, 0 otherwise). In **recommendation**, this connects to bandits and MDPs; policy gradient methods can optimize for delayed effects (e.g. diversity, exploration).

**Where you see this in practice:** RL for recommendation (YouTube, etc.); bandits and MDPs for sequential recommendation; long-term engagement optimization.

**Illustration (recommender engagement):** An RL agent maximizes long-term clicks; cumulative reward per episode (e.g. 20 steps) improves as the policy learns user preferences. The chart below shows reward per episode.

{{< chart type="line" palette="return" title="Cumulative reward per episode (recommender)" labels="0, 200, 400, 600, 800" data="2, 5, 8, 11, 14" xLabel="Episode" yLabel="Cumulative reward" >}}

**Exercise:** Build a toy recommender with 100 items and a user model that has changing preferences. Train an agent to maximize long-term user engagement (e.g., cumulative clicks). Use a policy gradient method.

**Professor's hints**

- **User model:** e.g. latent vector u_t that drifts: u_{t+1} = 0.9*u_t + 0.1*noise, or u depends on last K items (context). Probability of click for item i: P(click | u, i) = σ(u^T v_i) where v_i is item embedding. So "changing preferences" = u changes over time.
- **State:** Last L recommended items and outcomes (clicks), or summary; or the agent does not see u (partial obs) and only sees history. Action = which of 100 items to show. Reward = 1 if click, 0 else (or watch time).
- **Policy:** Output distribution over 100 items (or over a subset for efficiency). Use REINFORCE or PPO; episode = one user session (e.g. T = 20 steps). Return = sum of rewards in the session.
- **Baseline:** Random recommendation, or greedy (recommend item with highest estimated P(click) under current user model if you have access). Compare cumulative reward per episode.

**Common pitfalls**

- **Partial observability:** The agent may not see the user's true preference u; state is then history only. The policy must learn to infer or explore.
- **Cold start:** With 100 items and little data, many items get few clicks; use exploration (e.g. entropy bonus, or prior over items).
- **Reward design:** Click is a simple reward; in practice, long-term satisfaction may require diversity or novelty. You can add a small bonus for recommending less-seen items.

{{< collapse summary="Worked solution (warm-up: recommender as RL)" >}}
**Key idea:** Recommendation can be framed as RL: state = user context (history, profile), action = which item(s) to show, reward = click, watch time, or purchase. We want to maximize long-term engagement (return). Bandits are the single-step case; full RL handles delayed feedback (e.g. user returns later). We need a simulator or logged data (offline/batch RL) and a reward proxy (e.g. click) that correlates with true satisfaction.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why might we want to maximize long-term engagement instead of only the next click?
2. **Coding:** Implement the toy recommender: 100 items, user u_t with drift, P(click)=σ(u^T v_i). State = last 5 (item, click) pairs. Policy = MLP that outputs logits over 100 items. Train with REINFORCE for 1000 episodes (each episode = 20 steps). Plot cumulative reward per episode. Compare with random and greedy (argmax P(click) if u were known).
3. **Challenge:** Add **diversity** reward: small bonus for recommending items that are different from recently recommended (e.g. negative similarity to last K items). Does the policy learn to diversify and does long-term reward improve?
