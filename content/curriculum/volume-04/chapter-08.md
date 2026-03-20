---
title: "Chapter 38: Continuous Action Spaces"
description: "Policy network for Pendulum: Gaussian mean and log-std; log-prob."
date: 2026-03-10T00:00:00Z
weight: 38
draft: false
difficulty: 7
tags: ["continuous actions", "Pendulum", "Gaussian policy", "curriculum"]
keywords: ["continuous action spaces", "Gaussian policy", "log-prob", "Pendulum"]
roadmap_color: "amber"
roadmap_icon: "trend-up"
roadmap_phase_label: "Vol 4 · Ch 8"
---

**Learning objectives**

- Design a **policy network for continuous actions** that outputs the **mean** and **log-standard deviation** of a Gaussian (or similar) distribution.
- **Sample** actions from the distribution and compute **log-probability** \\(\log \pi(a|s)\\) for use in policy gradient updates.
- Apply this to an environment with continuous actions (e.g. **Pendulum-v1**).

**Concept and real-world RL**

For **continuous actions** (e.g. torque, throttle), we cannot use a softmax over a finite set. Instead we use a **continuous distribution**, often a **Gaussian**: \\(\pi(a|s) = \mathcal{N}(a; \mu(s), \sigma(s)^2)\\). The policy network outputs \\(\mu(s)\\) and \\(\log \sigma(s)\\) (log-std for stability); we sample \\(a = \mu + \sigma \cdot z\\) with \\(z \sim \mathcal{N}(0,1)\\). The log-probability is \\(\log \pi(a|s) = -\frac{1}{2}(\log(2\pi) + 2\log\sigma + \frac{(a-\mu)^2}{\sigma^2})\\). In **robot control** (e.g. Pendulum, MuJoCo), actions are continuous; the same Gaussian policy is used in REINFORCE, actor-critic, and PPO for continuous control.

**Where you see this in practice:** Gaussian policies are standard in continuous control (Pendulum, HalfCheetah, robotics simulators). Bounded actions are often handled by squashing (e.g. tanh) with a correction in the log-prob.

**Illustration (action distribution):** A Gaussian policy samples actions from \\(\\mathcal{N}(\\mu(s), \\sigma^2)\\). The chart below shows the distribution of 100 sampled actions (histogram bins) for a fixed state.

{{< chart type="bar" title="Action samples from π(·|s) (5 bins)" labels="Bin 1, Bin 2, Bin 3, Bin 4, Bin 5" data="8, 22, 40, 22, 8" >}}

**Exercise:** Design a policy network for continuous actions (e.g., Pendulum-v1) that outputs mean and log-std of a Gaussian. Write code to sample actions and compute log-probability for training.

**Professor's hints**

- Pendulum-v1: action is 1D in \\([-2, 2]\\). Output \\(\mu\\) (1 dim) and \\(\log \sigma\\) (1 dim). Clamp \\(\sigma\\) to a minimum (e.g. 1e-2) to avoid collapse. Sample: `a = mu + sigma * torch.randn(...)`.
- Log-prob: \\(\log \pi(a|s) = -0.5 (\\log(2\\pi) + 2\\log\\sigma + ((a-\\mu)/\\sigma)^2)\\). In PyTorch: use `torch.distributions.Normal(mu, sigma)` and `.log_prob(a)`.
- If the env expects actions in a bounded range, clip or use tanh: \\(a = \tanh(a_{raw})\\) and add the log-determinant correction for the gradient: \\(\log \pi(a|s) = \log \pi(a_{raw}|s) - \log(1 - a^2)\\) (for tanh squashing).

**Common pitfalls**

- **Sigma too small:** If \\(\sigma \to 0\\), the policy becomes almost deterministic and exploration stops. Use a lower bound on \\(\sigma\\) (e.g. 0.01) or a minimum log_std.
- **Wrong log-prob for bounded actions:** If you squash with tanh, the density on \\(a\\) is not the same as the density on \\(a_{raw}\\); you must add the Jacobian correction \\(-\sum \log(1 - a^2)\\) for tanh.

{{< collapse summary="Worked solution (warm-up: Gaussian log-prob)" >}}
**Warm-up:** For \\(\\mathcal{N}(0,1)\\), \\(\\log p(a) = -\\frac{1}{2}\\log(2\\pi) - \\frac{a^2}{2}\\). At \\(a=0.5\\): \\(\\log p(0.5) = -\\frac{1}{2}\\log(2\\pi) - 0.125 \\approx -1.04\\). Check: `scipy.stats.norm(0,1).logpdf(0.5)` gives the same. In continuous policy gradient we use \\(\\nabla \\log \\pi(a|s)\\) which for a Gaussian is \\(\\frac{(a - \\mu)}{\\sigma^2} \\nabla \\mu\\) (and similar for \\(\\sigma\\) if learned).
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For a 1D Gaussian with \\(\mu=0, \sigma=1\\), write the log-probability of \\(a=0.5\\) in closed form. Check with `scipy.stats.norm(0,1).logpdf(0.5)`.
2. **Coding:** Implement a Gaussian policy network for Pendulum: input 3-dim state, output \\(\mu\\) and \\(\log \\sigma\\). Sample 100 actions from the policy (with a fixed random state) and plot a histogram. Compute the mean log-prob of those samples.
3. **Challenge:** Add **tanh squashing** so the action is in \\((-1, 1)\\): \\(a = \tanh(a_{raw})\\). Derive the log-probability of \\(a\\) given \\(a_{raw} \\sim \mathcal{N}(\\mu, \\sigma^2)\\) (include the derivative of tanh). Implement it and use it in a short REINFORCE loop for Pendulum.
4. **Variant:** Try a very small \\(\\log \\sigma = -5\\) (deterministic) vs \\(\\log \\sigma = 0\\) (unit std). How does exploration change? Does the near-deterministic policy get stuck in local optima?

{{< pyrepl code="import torch\nimport torch.distributions as dist\n\n# Gaussian policy for 1D action\nmu = torch.tensor([0.0])\nfor log_sigma in [-2.0, 0.0, 1.0]:\n    sigma = torch.exp(torch.tensor(log_sigma))\n    p = dist.Normal(mu, sigma)\n    actions = p.sample((100,))\n    print(f'log_sigma={log_sigma:.1f}: mean={actions.mean():.2f}, std={actions.std():.2f}')" height="200" >}}

5. **Debug:** The code below forgets to add the tanh Jacobian correction to the log-probability, causing the gradient to be wrong for bounded actions. Fix it.

```python
def log_prob_squashed(mu, log_sigma, a_raw):
    sigma = log_sigma.exp()
    dist = torch.distributions.Normal(mu, sigma)
    lp = dist.log_prob(a_raw)
    # BUG: missing tanh correction
    # Fix: lp -= torch.log(1 - a_raw.tanh().pow(2) + 1e-6)
    return lp
```

6. **Conceptual:** Why do we output \\(\\log \\sigma\\) (log-standard deviation) instead of \\(\\sigma\\) directly? What numerical problem would arise if we used \\(\\sigma\\) without the log?
7. **Recall:** Write the log-probability formula for a Gaussian policy \\(\\log \\pi(a|s) = \\ldots\\) in terms of \\(\\mu, \\sigma, a\\) from memory.
