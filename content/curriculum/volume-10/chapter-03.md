---
title: "Chapter 93: RL for Algorithmic Trading"
description: "Simple stock MDP: buy/sell/hold; profit reward; Sharpe ratio."
date: 2026-03-10T00:00:00Z
weight: 93
draft: false
tags: ["algorithmic trading", "MDP", "Sharpe ratio", "trading", "curriculum"]
keywords: ["RL for trading", "algorithmic trading", "buy sell hold", "Sharpe ratio"]
---

**Learning objectives**

- **Simulate** a simple **stock market** with one asset (e.g. price follows a random walk or a simple mean-reverting process).
- **Design** an **MDP**: state = (price, position, cash, or features); actions = buy / sell / hold (possibly with size); reward = profit (or risk-adjusted return).
- **Train** an agent (e.g. DQN or PPO) on this MDP and **evaluate** its **Sharpe ratio** (mean return / std return over episodes or over time).
- **Discuss** **risk management**: position limits, drawdown, transaction costs; how the reward and state design affect behavior.
- **Relate** the exercise to **trading** and **finance** anchor scenarios (state = market + portfolio, action = trade, reward = profit or Sharpe).

**Concept and real-world RL**

**RL for algorithmic trading** models the agent as a trader: **state** includes market data (e.g. price, volume) and portfolio (position, cash); **actions** are trading decisions (buy/sell/hold or order size); **reward** is often profit or risk-adjusted return (e.g. Sharpe ratio). A simple **one-asset** market can be simulated with a random walk or a simple stochastic process. The agent must learn to exploit structure (e.g. mean reversion or trend) while managing risk. In **trading** and **finance**, RL is used for execution, portfolio allocation, and market making; evaluation typically includes Sharpe ratio and drawdown.

**Where you see this in practice:** RL for trading and execution; portfolio optimization; risk-sensitive RL.

**Illustration (trading reward):** In a simple MDP, the agent's profit (reward) over episodes can be volatile. The chart below shows cumulative profit over 20 episodes (example).

{{< chart type="line" palette="return" title="Cumulative profit over episodes" labels="0, 5, 10, 15, 20" data="0, 50, 120, 80, 150" xLabel="Episode" yLabel="Profit" >}}

**Exercise:** Simulate a simple stock market with one asset. Design an MDP where actions are buy/sell/hold, reward is profit. Train an agent and evaluate its Sharpe ratio. Discuss risk management.

**Professor's hints**

- **Market sim:** e.g. log price p_t = p_{t-1} + ε_t, ε_t ~ N(0, σ^2), or mean-reverting. One asset; maybe add bid-ask spread or transaction cost (e.g. 0.1% per trade).
- **MDP:** State = (price, position, cash) or (returns, position). Actions: discrete {buy, sell, hold} or continuous (order size). Reward = change in portfolio value (or daily PnL) minus transaction costs. Use a finite horizon (e.g. 100 steps per episode).
- **Training:** DQN or PPO; collect episodes and update. Reward shaping: raw profit is fine; optionally use risk-adjusted (e.g. reward = return - β * variance) to encourage stability.
- **Sharpe ratio:** Over N episodes (or N windows), compute mean return R and std σ; Sharpe ≈ R / σ (annualized if you use daily returns: multiply by sqrt(252) or similar). Report mean return, std, and Sharpe.
- **Risk management:** Discuss in 2–3 sentences: e.g. position limits (cap |position|), stop-loss (exit if drawdown > X), or reward that penalizes variance.

**Common pitfalls**

- **Overfitting:** A simple market sim may be too easy or too noisy; the agent might overfit to the sim. Use different seeds or out-of-sample periods for evaluation.
- **Transaction costs:** If ignored, the agent may trade too often; include costs in the reward.
- **Sharpe on training data:** Report Sharpe on a **held-out** period or different random seeds to avoid overfitting.

{{< collapse summary="Worked solution (warm-up: RL in finance)" >}}
**Key idea:** RL can be used for trading: state = market features (or portfolio state), action = trade (buy/sell/hold or weights), reward = profit or risk-adjusted return (e.g. Sharpe). We must avoid overfitting: use held-out periods, multiple seeds, and simple policies. The environment is non-stationary and noisy; so sample efficiency and robustness matter. Common choices: PPO or DDPG with careful reward design (e.g. transaction costs, drawdown penalty).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Why is the Sharpe ratio a useful metric for trading strategies compared to raw profit?
2. **Coding:** Implement a 1-asset market (random walk), MDP with buy/sell/hold, and train DQN for 500 episodes. Evaluate on 100 new episodes; report mean return, std, Sharpe. Add 0.1% transaction cost and retrain; does the policy trade less?
3. **Challenge:** Add a **drawdown** constraint: penalize reward when portfolio value drops more than 10% from peak. Train with this penalty and compare Sharpe and max drawdown with the unconstrained agent.
4. **Variant:** Change the market model from a random walk to a mean-reverting process. Does the DQN policy change strategy (e.g. trade more frequently)? Compare Sharpe ratios across market models.
5. **Debug:** A trading DQN achieves high returns during training (seen market data) but near-zero Sharpe on the held-out test set. The agent has overfit to specific price patterns in training data. Describe how to implement proper temporal train/validation/test splits for financial time-series and why shuffling the replay buffer across time steps violates this.
6. **Conceptual:** Financial RL faces a unique challenge: the agent's own actions affect market prices (market impact). Explain how this feedback loop violates the standard MDP assumption of an agent not affecting environment dynamics, and describe one mitigation strategy used in practice.
