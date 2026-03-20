---
title: "Statistics for RL"
description: "Mean, variance, standard deviation, standard error, histograms, and correlation — with RL motivation and practice."
date: 2026-03-20T00:00:00Z
draft: false
difficulty: 3
tags: ["statistics", "mean", "variance", "standard deviation", "math for RL"]
keywords: ["statistics for RL", "mean", "variance", "standard deviation", "standard error", "correlation"]
weight: 2
roadmap_icon: "chart"
roadmap_color: "teal"
roadmap_phase_label: "Phase 3 · Statistics"
---

This page covers the statistics you need to analyze RL experiments: computing and interpreting mean, variance, standard deviation, standard error, histograms, and correlation. [Back to Math for RL](../).

---

## Core concepts

### Mean

The **sample mean** (average) of observations \\(x_1, \ldots, x_n\\) is: \\(\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i\\)

The mean is the single best summary of a dataset — it tells you the center of the distribution.

**In RL:** The average reward per episode is the most common performance metric. When comparing two policies, you compare their mean episode rewards across many evaluation runs.

---

### Variance and standard deviation

The **sample variance** measures spread around the mean:
\\(\sigma^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2\\)

The **standard deviation** is \\(\sigma = \sqrt{\sigma^2}\\) — it is in the same units as the data, making it more interpretable than variance.

**In RL:** High variance in returns means the policy is **unstable** — it sometimes achieves high rewards and sometimes fails. An unstable policy (high \\(\sigma\\)) is often worse than one with a lower mean but more consistency. Policy gradient methods (like PPO) explicitly reduce variance in the gradient estimate by using advantage functions.

{{< chart type="bar" palette="math" title="Episode returns showing high variance" labels="Ep 1,Ep 2,Ep 3,Ep 4,Ep 5,Ep 6" data="12,45,8,50,5,42" yLabel="Return" >}}

---

### Standard error of the mean

The **standard error** (SE) measures uncertainty in the sample mean itself:
\\(\text{SE} = \frac{\sigma}{\sqrt{n}}\\)

SE decreases as you collect more samples — the more evaluation runs you do, the more confident you are in the mean.

**In RL:** When reporting evaluation results, always report mean ± SE (or mean ± std) over multiple evaluation episodes. A result like "mean reward = 250 ± 15" is meaningful; "mean reward = 250" from 2 episodes is not.

---

### Histograms

A **histogram** shows the distribution of values by counting how many observations fall into each bin. It reveals shape (unimodal, bimodal, skewed), outliers, and spread.

**In RL:** Plotting the histogram of episode returns over many evaluation runs shows whether the policy has a consistent distribution or occasional catastrophic failures. A bimodal histogram (returns cluster near 0 and near 500) suggests the policy sometimes gets stuck.

{{< chart type="bar" palette="math" title="Histogram of episode returns (100 episodes)" labels="0–50,50–100,100–150,150–200,200–250" data="5,15,30,35,15" yLabel="Count" >}}

---

### Correlation

The **Pearson correlation coefficient** \\(r\\) measures the linear relationship between two variables:

\\(r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i-\bar{x})^2 \cdot \sum_{i=1}^n (y_i-\bar{y})^2}}\\)

\\(r \in [-1, 1]\\): \\(r=1\\) means perfect positive correlation, \\(r=-1\\) perfect negative, \\(r=0\\) no linear relationship.

**In RL:** Correlation between reward and episode length can reveal whether "longer is better" (e.g. in survival tasks) or "shorter is better" (e.g. solve the maze faster). Understanding correlations in your data helps diagnose agent behavior.

{{< chart type="line" palette="math" title="Episode length vs reward (positive correlation r ≈ 0.85)" labels="10,20,30,40,50,60,70,80" data="25,45,55,70,80,90,100,110" xLabel="Episode length" yLabel="Reward" >}}

---

## Practice questions

1. **Compute by hand:** Episode returns: [10, 20, 30, 40, 50]. Compute the sample mean and unbiased sample variance.

{{< collapse summary="Answer" >}}
**Mean:** \\(\bar{x} = (10+20+30+40+50)/5 = 150/5 = 30\\).

**Variance:** Deviations from mean: \\(-20, -10, 0, 10, 20\\). Squared: \\(400, 100, 0, 100, 400\\). Sum = 1000. Unbiased variance: \\(1000/(5-1) = 250\\).

**Std:** \\(\sqrt{250} \approx 15.81\\).

**In RL:** A mean return of 30 with std ≈ 15.8 means about 68% of episodes return between 14 and 46 (±1 std for a normal distribution).
{{< /collapse >}}

The chart below shows the five returns and their mean (30).

{{< chart type="bar" palette="math" title="Episode returns and mean (30)" labels="Ep1,Ep2,Ep3,Ep4,Ep5" data="10,20,30,40,50" yLabel="Return" >}}

---

2. **NumPy:** Compute mean and standard deviation for a rewards list using NumPy.

{{< pyrepl code="import numpy as np\n\nrewards = [10, 20, 30, 40, 50]\nmean = np.mean(rewards)\nstd = np.std(rewards, ddof=1)  # ddof=1 for unbiased (n-1)\nprint(f'Mean: {mean}')\nprint(f'Std (unbiased): {std:.4f}')\nprint(f'Min: {min(rewards)}, Max: {max(rewards)}')\nprint(f'Median: {np.median(rewards)}')" height="180" >}}

{{< collapse summary="Answer" >}}
`np.mean` computes the arithmetic mean. `np.std(ddof=1)` uses the unbiased \\(n-1\\) denominator (same as what you'd compute by hand for sample std). `ddof=0` would give the biased version. Always use `ddof=1` when estimating population std from a sample.
{{< /collapse >}}

---

3. **Standard error:** Three evaluation runs give mean episode rewards: [240, 260, 250]. Compute the standard error of the mean.

{{< collapse summary="Answer" >}}
**Step 1 — mean:** \\(\bar{x} = (240+260+250)/3 = 250\\).

**Step 2 — deviations:** \\(-10, 10, 0\\). Squared: \\(100, 100, 0\\). Sum = 200.

**Step 3 — std:** \\(\sigma = \sqrt{200/2} = \sqrt{100} = 10\\).

**Step 4 — SE:** \\(\text{SE} = \sigma/\sqrt{n} = 10/\sqrt{3} \approx 5.77\\).

**Report:** Mean reward = 250 ± 5.77 (SE). This tells us our estimate of the true mean could easily be off by ~6 reward units.
{{< /collapse >}}

---

4. **Standard error in NumPy:** Compute standard error across 10 evaluation episodes.

{{< pyrepl code="import numpy as np\n\n# 10 evaluation episodes\nepisode_returns = [230, 245, 255, 240, 270, 235, 260, 250, 265, 248]\n\nmean = np.mean(episode_returns)\nstd = np.std(episode_returns, ddof=1)\nn = len(episode_returns)\nse = std / np.sqrt(n)\n\nprint(f'Mean:           {mean:.1f}')\nprint(f'Std:            {std:.2f}')\nprint(f'Std Error (SE): {se:.2f}')\nprint(f'Report: {mean:.1f} \\u00b1 {se:.2f} (SE)')" height="200" >}}

{{< collapse summary="Answer" >}}
Standard error = std / sqrt(n). With 10 episodes and std ≈ 13.4, SE ≈ 4.2. The reported mean ± SE gives a meaningful confidence interval: roughly 2×SE covers ≈95% of plausible true means (for large n). With only 3–5 evaluation episodes, SE is large — always use at least 10–30 evaluation episodes for reliable results.
{{< /collapse >}}

---

5. **By hand — std interpretation:** Policy A: mean = 200, std = 5. Policy B: mean = 210, std = 80. Which policy is more reliable? Which would you prefer for a safety-critical application?

{{< collapse summary="Answer" >}}
**Policy A** is more reliable: lower variance (std=5) means it consistently achieves around 200.

**Policy B** has a higher mean but massive variance (std=80): it sometimes gets much higher rewards but also sometimes fails badly.

For safety-critical applications (medical robots, self-driving), always prefer low variance. For high-stakes competitive settings, you might accept higher variance for the higher mean.

In RL research, always report both mean and std (or SE) — a result claiming "200 mean reward" is incomplete without uncertainty.
{{< /collapse >}}

---

6. **Histogram interpretation:** An RL agent's episode returns histogram has two peaks: one near 0 (50% of episodes) and one near 500 (50% of episodes). The mean is 250. Is this agent performing well?

{{< collapse summary="Answer" >}}
No. The mean of 250 is **misleading** — the agent never actually achieves around 250. It either succeeds (≈500) or completely fails (≈0). This **bimodal distribution** reveals an unstable policy.

A histogram is essential here: looking only at the mean would suggest moderate performance, hiding the binary success/failure behavior. In practice this might mean the agent's policy is at a tipping point — small changes in environment cause drastically different outcomes.
{{< /collapse >}}

{{< chart type="bar" palette="math" title="Bimodal return distribution (mean=250 is misleading)" labels="0–100,100–200,200–300,300–400,400–500" data="45,3,2,4,46" yLabel="Count (out of 100 episodes)" >}}

---

7. **Correlation:** Episode rewards: [100, 150, 200, 250, 300] and episode lengths: [10, 15, 20, 25, 30]. Compute correlation by inspection. What does this tell you?

{{< collapse summary="Answer" >}}
Both sequences increase perfectly together in equal steps — this is **perfect positive correlation** (\\(r=1\\)). In this task, longer episodes always mean higher rewards (e.g. a survival task where the agent earns +1 per step).

**Implication:** You could track episode length as a proxy for reward. Correlation analysis helps identify redundant metrics and understand the structure of your RL problem.
{{< /collapse >}}

---

8. **Extra — Correlation in NumPy:** Compute Pearson correlation between reward and episode length for the data above.

{{< pyrepl code="import numpy as np\n\nrewards = np.array([100, 150, 200, 250, 300], dtype=float)\nlengths = np.array([10, 15, 20, 25, 30], dtype=float)\n\n# Pearson correlation\ndef pearson_r(x, y):\n    xm = x - x.mean()\n    ym = y - y.mean()\n    return np.sum(xm * ym) / np.sqrt(np.sum(xm**2) * np.sum(ym**2))\n\nr = pearson_r(rewards, lengths)\nprint(f'Pearson r: {r:.4f}')  # expected: 1.0\n# Also using numpy:\nprint(f'np.corrcoef: {np.corrcoef(rewards, lengths)[0,1]:.4f}')" height="220" >}}

{{< collapse summary="Answer" >}}
`np.corrcoef(x, y)[0, 1]` returns the off-diagonal element of the 2×2 correlation matrix — that is the correlation between x and y. For perfectly linear data the result is exactly 1.0.

Note: correlation only measures **linear** relationships. Two variables can have strong non-linear relationships but \\(r \approx 0\\).
{{< /collapse >}}

---

9. **Extra — Plot histogram:** Plot the histogram of 100 episode returns sampled from a normal distribution with mean=250, std=40.

{{< pyrepl code="import numpy as np\nimport matplotlib.pyplot as plt\n\nnp.random.seed(42)\nreturns = np.random.normal(loc=250, scale=40, size=100)\n\nplt.figure(figsize=(7, 4))\nplt.hist(returns, bins=15, color='steelblue', edgecolor='white', alpha=0.85)\nplt.axvline(returns.mean(), color='red', linestyle='--', label=f'mean={returns.mean():.1f}')\nplt.xlabel('Episode return')\nplt.ylabel('Count')\nplt.title('Distribution of episode returns')\nplt.legend()\nplt.tight_layout()\nplt.savefig('returns_hist.png', dpi=80)\nplt.show()\nprint(f'Mean: {returns.mean():.2f}, Std: {returns.std(ddof=1):.2f}, SE: {returns.std(ddof=1)/np.sqrt(len(returns)):.2f}')" height="260" >}}

---

10. **Extra — Which metric?** You have 5 evaluation episodes with returns [50, 55, 52, 48, 60]. Your colleague has 100 evaluation episodes with mean 52, std 15. Whose estimate of the true mean is more reliable? Compute both standard errors.

{{< collapse summary="Answer" >}}
**Your SE:** std([50,55,52,48,60]) = sqrt(((0)^2+25+4+16+64)/4) = sqrt(109/4) ≈ 5.22. SE = 5.22/sqrt(5) ≈ **2.33**.

**Colleague's SE:** std=15, n=100. SE = 15/sqrt(100) = **1.5**.

Your colleague's SE (1.5) is smaller despite a higher std, because they have 20× more samples. Their mean estimate is more reliable.

**Key lesson:** SE shrinks with \\(1/\sqrt{n}\\). To halve SE, you need 4× more evaluation episodes.
{{< /collapse >}}

---

11. **Extra — Variance in RL training:** During DQN training, you log the loss every 100 steps. The first 1000 steps show loss variance = 40; the next 1000 show variance = 5. What does this tell you about training progress?

{{< collapse summary="Answer" >}}
Decreasing loss variance during training indicates **stabilization**: the Q-network is converging to more consistent predictions. Early training is volatile (high variance loss) because the Q-values are changing rapidly; later training is stable. If variance stays high indefinitely, the learning rate may be too large or the replay buffer too small (correlated samples).
{{< /collapse >}}

---

12. **Extra — Z-score:** An episode return of 350 has mean 250 and std 50. What is the z-score? What does it mean?

{{< collapse summary="Answer" >}}
**Z-score:** \\(z = (x - \mu)/\sigma = (350 - 250)/50 = \mathbf{2.0}\\).

This means the return of 350 is **2 standard deviations above the mean** — it's in approximately the top 2.3% of returns (for a normal distribution). Z-scores standardize different metrics to the same scale, useful when comparing across environments with different reward ranges.
{{< /collapse >}}

---

13. **Extra — Outlier detection:** Returns: [240, 250, 245, 255, 248, 2]. The last value looks like an outlier. Compute mean and median with and without it. Which statistic is more robust to outliers?

{{< collapse summary="Answer" >}}
**With outlier:** mean = (240+250+245+255+248+2)/6 = 1240/6 ≈ 206.7. Median = (245+248)/2 = 246.5.

**Without outlier:** mean = (240+250+245+255+248)/5 = 247.6. Median = 248.

**Median** is much more robust — it barely changes. The mean drops dramatically from 247.6 to 206.7 due to one outlier. In RL, plot the distribution and consider using median episode return for noisy environments.
{{< /collapse >}}

---

14. **Extra — Sample size planning:** You want SE ≤ 5 for an RL evaluation where std ≈ 30. How many evaluation episodes do you need?

{{< collapse summary="Answer" >}}
\\(\text{SE} = \sigma/\sqrt{n} \leq 5\\) → \\(\sqrt{n} \geq 30/5 = 6\\) → \\(n \geq 36\\).

You need at least **36 evaluation episodes**. This is a practical calculation every RL practitioner should do before deciding how many evaluation episodes to run.

**Python:** `import math; n = math.ceil((30/5)**2); print(n)` → 36.
{{< /collapse >}}

---

15. **Extra — Bootstrapped confidence interval:** You have 10 episode returns. Describe the bootstrapping procedure for computing a 95% confidence interval on the mean. (No pyrepl needed — conceptual.)

{{< collapse summary="Answer" >}}
**Bootstrap procedure:**
1. From your 10 returns, sample 10 values **with replacement** (some values may appear multiple times, others not at all). This is one bootstrap sample.
2. Compute the mean of this bootstrap sample.
3. Repeat steps 1–2 many times (e.g. 1000 times) to get 1000 bootstrap means.
4. The 2.5th and 97.5th percentiles of these 1000 means form the **95% bootstrap CI**.

This works without assuming normality — useful when you have few episodes or heavy-tailed return distributions. In RL research, bootstrap CIs are more honest than ±1 SE when sample sizes are small.
{{< /collapse >}}

---

## Professor's hints

- Always use `ddof=1` in NumPy (`np.std(x, ddof=1)`) when computing sample std from data — this gives the unbiased estimator.
- **Report SE, not std, for the uncertainty in the mean.** Std tells you about the spread of individual episodes; SE tells you about uncertainty in your estimate of the mean performance.
- Before comparing two policies, check if the difference in means is larger than 2× the SE of each — otherwise the difference may not be significant.
- A histogram of episode returns is the most informative single plot for understanding an RL agent's behavior. Never trust just the mean.

---

## Common pitfalls

- **Reporting std when you mean SE:** Std ≈ 40 looks alarming; SE ≈ 4 (with n=100) looks fine. They measure different things. SE = std/sqrt(n).
- **Too few evaluation episodes:** With n=3, SE is huge — any comparison is meaningless. Use at least 10–30 evaluation episodes.
- **Confusing sample mean with expected value:** The sample mean is an estimate; the true expected return is a fixed property of the policy. They converge as n→∞ (law of large numbers).
- **Ignoring the distribution shape:** Mean and std don't distinguish unimodal from bimodal distributions. Always plot the histogram.
