---
title: "Bandits: Thompson Sampling"
description: "Bayesian bandits and Thompson Sampling—sample from the posterior to balance exploration and exploitation."
date: 2026-03-10T00:00:00Z
weight: 5
draft: false
tags: ["bandits", "Thompson sampling", "Bayesian", "curriculum"]
keywords: ["Thompson sampling", "Bayesian bandits", "posterior", "Beta Bernoulli"]
---

**Learning objectives**

- Understand the Bayesian view: maintain a posterior over each arm’s reward distribution.
- Implement Thompson Sampling for Bernoulli and Gaussian rewards.
- Compare Thompson Sampling with epsilon-greedy and UCB1.

## Theory (pt 1): Bernoulli bandits

Suppose each arm gives a reward 0 or 1 (e.g. click or no click). We model arm \\(a\\) as Bernoulli with unknown mean \\(\theta_a\\). A convenient prior is **Beta**: \\(\theta_a \sim \text{Beta}(\alpha_a, \beta_a)\\). After observing \\(s\\) successes and \\(f\\) failures from arm \\(a\\), the posterior is \\(\text{Beta}(\alpha_a + s, \beta_a + f)\\).

**Thompson Sampling:** For each arm, **sample** \\(\tilde{\theta}_a\\) from its posterior. Choose the arm with the largest \\(\tilde{\theta}_a\\). Then pull that arm, observe the reward, and update that arm’s posterior. This naturally balances exploration (uncertain arms have high variance in the sample) and exploitation (arms with high posterior mean tend to get high samples).

## Theory (pt 2): Gaussian rewards

For Gaussian rewards with unknown mean \\(\mu_a\\) and known variance \\(\sigma^2\\), we use a **normal prior** on \\(\mu_a\\). With a normal likelihood, the posterior for \\(\mu_a\\) is also normal (conjugate prior). Thompson Sampling: sample \\(\tilde{\mu}_a\\) from the posterior of each arm, then choose \\(a = \arg\max_a \tilde{\mu}_a\\), pull, observe reward, and update the posterior for that arm.

Details: Prior \\(\mu_a \sim \mathcal{N}(m_0, v_0)\\). After \\(n\\) observations with sample mean \\(\bar{x}\\) and variance \\(\sigma^2/n\\), the posterior mean and variance have closed form. Sampling from that posterior gives \\(\tilde{\mu}_a\\); we pick \\(\arg\max_a \tilde{\mu}_a\\).

## Beginner's exercise prompt

1. **Bernoulli:** Implement Thompson Sampling for 10 Bernoulli arms (each with unknown \\(p_a \in [0,1]\\)). Use Beta(1,1) prior. Run 1000 steps, average over runs, plot average reward. Compare with epsilon-greedy.
2. **Gaussian:** Implement Thompson Sampling for the 10-armed Gaussian testbed (unknown mean, known variance 1). Use a normal prior (e.g. mean 0, variance 1). Run 1000 steps and compare with UCB1 and epsilon-greedy.

## Code sketch (Gaussian)

- For each arm maintain posterior mean \\(m_a\\) and variance \\(v_a\\) (or precision). Start with prior \\(m_a = 0\\), \\(v_a = 1\\) (or your choice).
- At each step: for each arm, sample \\(\tilde{\mu}_a \sim \mathcal{N}(m_a, v_a)\\). Choose \\(a = \arg\max_a \tilde{\mu}_a\\).
- Pull arm \\(a\\), get reward \\(r\\). Update arm \\(a\\)’s posterior using the Gaussian conjugate update (posterior mean and variance given new observation).

**Exercise on Gaussian rewards:** Try different priors (e.g. optimistic prior mean) and see how quickly Thompson Sampling converges compared to UCB1.

See [Chapter 2: Multi-Armed Bandits](chapter-02/) for the 10-armed testbed and [Bandits: UCB1](bandits-ucb1/) for a non-Bayesian alternative.
