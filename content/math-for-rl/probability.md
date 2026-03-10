---
title: "Probability & Statistics"
description: "Expectations, variance, sample mean, distributions, and law of large numbers — with RL motivation and practice."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["probability", "statistics", "expectation", "variance", "math for RL"]
keywords: ["probability for RL", "expectation", "variance", "sample mean", "law of large numbers"]
---

This page covers the probability and statistics you need for RL: expectations, variance, sample means, and the idea that sample averages converge to expectations. [Back to Math for RL](../).

---

## Core concepts

### Random variables and expectation

A **random variable** \\(X\\) takes values according to some distribution. The **expected value** (or **expectation**) \\(\mathbb{E}[X]\\) is the long-run average if you repeat the experiment infinitely many times.

- For a discrete \\(X\\) with outcomes \\(x_i\\) and probabilities \\(p_i\\): \\(\mathbb{E}[X] = \sum_i x_i p_i\\).
- For a continuous distribution with density \\(p(x)\\): \\(\mathbb{E}[X] = \int x\,p(x)\,dx\\) (you will mostly see discrete or simple continuous cases in RL).

**In reinforcement learning:** The return (sum of discounted rewards) is a random variable because rewards and transitions can be random. The **value function** \\(V(s)\\) is the *expected* return from state \\(s\\). Multi-armed bandits: each arm has an *expected* reward; we estimate it from samples.

### Variance and sample variance

The **variance** of \\(X\\) is \\(\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]\\); it measures spread. The **sample variance** of \\(n\\) observations \\(x_1,\ldots,x_n\\) is often computed with \\(n-1\\) in the denominator (unbiased): \\(\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2\\), where \\(\bar{x} = \frac{1}{n}\sum_i x_i\\) is the **sample mean**.

**In reinforcement learning:** We use sample means to estimate expected rewards (e.g. per arm in a bandit) and sample variances to measure uncertainty. Monte Carlo methods use the *sample return* (average of rewards along a trajectory) to estimate the expected return.

### Law of large numbers

As the number of samples \\(n\\) grows, the **sample average** \\(\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i\\) converges to the expected value \\(\mathbb{E}[X]\\) (under mild conditions). So we can estimate unknown expectations by averaging many observations.

**In reinforcement learning:** We cannot compute true \\(V(s)\\) in one shot; we run many episodes and average returns to get Monte Carlo estimates. Bandit algorithms use sample means of rewards per arm to estimate which arm is best.

**Illustration (sample mean → expectation):** As the number of samples grows, the sample average tends to get closer to the expected value. The chart below shows a typical trend: sample mean (e.g. from a bandit arm with true mean 0.4) over different sample sizes. With more pulls, the estimate stabilizes near the true mean.

{{< chart type="line" palette="math" title="Sample mean vs number of samples" labels="5, 10, 20, 50, 100, 200" data="0.32, 0.36, 0.38, 0.39, 0.40, 0.40" xLabel="n (samples)" yLabel="Sample mean" >}}

### Distributions you will see

- **Normal (Gaussian)** \\(\mathcal{N}(\mu, \sigma^2)\\): mean \\(\mu\\), variance \\(\sigma^2\\). Used for noise, rewards, and many continuous quantities.
- **Bernoulli:** 0 or 1 with probability \\(p\\) and \\(1-p\\). Used for binary outcomes (e.g. success/fail, left/right).

---

## Practice questions

1. **Bandit:** An arm has true expected reward 1. You get 4 samples: 1.2, 0.8, 1.0, 1.4. What is the sample mean? What is the (unbiased) sample variance?

{{< collapse summary="Answer and explanation" >}}
**Step 1 — Sample mean:** \\(\bar{x} = \frac{1}{n}\sum_i x_i = \frac{1.2 + 0.8 + 1.0 + 1.4}{4} = \frac{4.4}{4} = 1.1\\).

**Step 2 — Deviations from mean:** \\(x_i - \bar{x}\\) are \\(0.1, -0.3, -0.1, 0.3\\). Squared: \\(0.01, 0.09, 0.01, 0.09\\). Sum = \\(0.20\\).

**Step 3 — Unbiased sample variance:** \\(\frac{1}{n-1}\sum (x_i - \bar{x})^2 = \frac{0.20}{3} = 0.0\overline{6}\\) (or about 0.067).

**Answer:** Sample mean = **1.1**; unbiased sample variance ≈ **0.067**.

**Explanation:** The sample mean 1.1 is our estimate of the arm’s expected reward (true value 1). We use \\(n-1\\) in the denominator so that on average the sample variance equals the true variance of the distribution. In bandits we use these to compare arms and to quantify uncertainty.

**Python:** `x = [1.2, 0.8, 1.0, 1.4]; mean = sum(x)/len(x); var = sum((xi-mean)**2 for xi in x)/(len(x)-1); print(mean, var)` gives 1.1 and about 0.067.
{{< /collapse >}}

The chart below shows these four sample values; the mean 1.1 is the average of the bar heights.

{{< chart type="bar" palette="math" title="Sample values (mean = 1.1)" labels="Sample 1, Sample 2, Sample 3, Sample 4" data="1.2, 0.8, 1.0, 1.4" yLabel="Value" >}}

---

2. **Concept:** In one sentence, what is the difference between \\(\mathbb{E}[X]\\) and the sample average of 100 draws from \\(X\\)? When do they coincide (in the limit)?

{{< collapse summary="Answer and explanation" >}}
\\(\mathbb{E}[X]\\) is the theoretical long-run average (a fixed number determined by the distribution). The sample average of 100 draws is the average of one finite set of observations and is random. They coincide in the limit: as the number of draws \\(n \to \infty\\), the sample average converges to \\(\mathbb{E}[X]\\) (law of large numbers).

**Explanation:** In RL we never know true expectations (e.g. true \\(V(s)\\) or true arm means). We estimate them from samples; the law of large numbers justifies why averaging many returns or rewards gives a good estimate.

**Python:** `np.random.seed(42); draws = np.random.randn(100); print(draws.mean(), np.mean(draws))` — sample mean of 100 draws; as \\(n\\) grows, it approaches \\(\mathbb{E}[X]\\).
{{< /collapse >}}

The chart below illustrates convergence: sample mean approaches the true expectation as \\(n\\) increases (same idea as in Core concepts).

{{< chart type="line" palette="math" title="Sample mean → E[X] as n grows" labels="n=10, n=50, n=100, n=200" data="0.1, 0.05, 0.02, 0.01" xLabel="n" yLabel="|error|" >}}

---

3. **By hand:** For observations [0, 1, 2, 3, 4], compute the sample mean and the unbiased sample variance.

{{< collapse summary="Answer and explanation" >}}
**Step 1 — Sample mean:** \\(\bar{x} = \frac{0+1+2+3+4}{5} = \frac{10}{5} = 2\\).

**Step 2 — Squared deviations:** \\((0-2)^2 = 4\\), \\((1-2)^2 = 1\\), \\((2-2)^2 = 0\\), \\((3-2)^2 = 1\\), \\((4-2)^2 = 4\\). Sum = \\(4+1+0+1+4 = 10\\).

**Step 3 — Unbiased variance:** \\(\frac{1}{n-1}\sum (x_i - \bar{x})^2 = \frac{10}{4} = 2.5\\).

**Answer:** Sample mean = **2**; unbiased sample variance = **2.5**.

**Explanation:** With \\(n=5\\) we divide by 4 so the variance estimate is unbiased. This is the same formula we use for reward variance in bandits or return variance in Monte Carlo.

**Python:** `x = [0,1,2,3,4]; mean = sum(x)/len(x); var = sum((xi-mean)**2 for xi in x)/(len(x)-1); print(mean, var)` → 2 and 2.5.
{{< /collapse >}}

The chart below shows the five observations and their mean (2); the spread is captured by the sample variance 2.5.

{{< chart type="bar" palette="math" title="Observations [0,1,2,3,4] and mean 2" labels="0, 1, 2, 3, 4" data="0, 1, 2, 3, 4" yLabel="Value" >}}

---

4. **Python:** Write a function `sample_mean(x)` that takes a list of numbers and returns their average. Then write `sample_variance(x)` that returns the unbiased variance. Test with [1, 2, 3, 4, 5].

{{< collapse summary="Answer and explanation" >}}
**Step 1 — sample_mean:** Sum the list and divide by length. **Step 2 — sample_variance:** Compute mean, then \\(\frac{1}{n-1}\sum (x_i - \bar{x})^2\\).

```python
def sample_mean(x):
    return sum(x) / len(x)

def sample_variance(x):
    n = len(x)
    mean = sample_mean(x)
    squared_deviations = [(xi - mean) ** 2 for xi in x]
    return sum(squared_deviations) / (n - 1)

# Test with [1, 2, 3, 4, 5]
data = [1, 2, 3, 4, 5]
print(sample_mean(data))      # 3.0
print(sample_variance(data))  # 2.5
```

**Explanation:** We use \\(n-1\\) so the expected value of this statistic equals the population variance. In RL you’ll do similar operations on reward batches or returns.
{{< /collapse >}}

Running the code on [1,2,3,4,5] gives mean 3 and variance 2.5. The chart below shows the five values and their mean.

{{< chart type="bar" palette="math" title="Data [1,2,3,4,5], mean = 3" labels="1, 2, 3, 4, 5" data="1, 2, 3, 4, 5" yLabel="Value" >}}

---

5. **RL:** Why do we need many episodes in Monte Carlo prediction to get a good value estimate? Relate your answer to the law of large numbers.

{{< collapse summary="Answer and explanation" >}}
\\(V(s)\\) is the *expected* return from state \\(s\\); we don’t have the distribution, only samples (returns from episodes that visit \\(s\\)). We estimate \\(V(s)\\) by the *sample average* of those returns. The law of large numbers says this sample average converges to the expectation as the number of episodes (samples) increases. So we need many episodes so that our average is close to the true \\(V(s)\\).

**Explanation:** With few episodes, the estimate is noisy; with many, it stabilizes. This is the same idea as estimating a bandit arm’s mean by averaging many pulls.

**Python:** Simulate MC: collect returns from state \\(s\\) over many episodes; `V_s = np.mean(returns_from_s)`. As N grows, V_s stabilizes.
{{< /collapse >}}

The chart below shows a typical trend: value estimate (e.g. \\(V(s)\\)) stabilizes as the number of episodes increases.

{{< chart type="line" palette="math" title="V(s) estimate vs number of episodes" labels="100, 500, 1k, 5k, 10k" data="0.5, 0.35, 0.22, 0.08, 0.03" xLabel="Episodes" yLabel="|error|" >}}

---

6. **By hand:** For a Bernoulli with \\(p = 0.3\\), what is \\(\mathbb{E}[X]\\) and \\(\mathrm{Var}(X)\\)? (E[X]=0.3, Var(X)=p(1-p)=0.21.)

{{< collapse summary="Answer and explanation" >}}
**Step 1 — Expectation:** For Bernoulli, \\(\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1-p) = p = 0.3\\).

**Step 2 — Variance:** \\(\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2\\). For Bernoulli, \\(X^2 = X\\), so \\(\mathbb{E}[X^2] = p\\). Thus \\(\mathrm{Var}(X) = p - p^2 = p(1-p) = 0.3 \times 0.7 = 0.21\\).

**Answer:** \\(\mathbb{E}[X] = 0.3\\); \\(\mathrm{Var}(X) = 0.21\\).

**Explanation:** Bernoulli is 0/1 with probability \\(p\\) and \\(1-p\\). This formula appears when we model binary outcomes (e.g. success/fail, left/right) in RL.

**Python:** `p = 0.3; E_X = p; Var_X = p*(1-p); print(E_X, Var_X)` → 0.3 and 0.21.
{{< /collapse >}}

For Bernoulli(\\(p=0.3\\)), \\(\mathbb{E}[X]=0.3\\) and \\(\mathrm{Var}(X)=0.21\\). The chart below shows the two outcomes and their probabilities.

{{< chart type="bar" palette="math" title="Bernoulli(0.3): P(X=0) and P(X=1)" labels="P(0), P(1)" data="0.7, 0.3" yLabel="P(X)" >}}

---

7. **RL:** In a bandit, we estimate \\(Q(a)\\) by the sample mean of rewards from arm \\(a\\). Why might we prefer this over using only the last reward from arm \\(a\\)?

{{< collapse summary="Answer and explanation" >}}
The sample mean uses *all* rewards observed from arm \\(a\\), so it has lower variance than a single reward. One reward is noisy and can be far from the true expected reward; the sample mean averages out the noise and, by the law of large numbers, converges to \\(Q(a)\\). Using only the last reward would ignore previous information and give a much noisier estimate.

**Explanation:** In bandit algorithms we maintain a running mean (or equivalent) per arm. That’s exactly the sample mean of all rewards from that arm so far. The last reward alone is an unbiased but high-variance estimate.

**Python:** `rewards_from_a = []; rewards_from_a.append(r); Q_a = sum(rewards_from_a)/len(rewards_from_a)` — running sample mean. Variance of this estimate decreases as more rewards are added.
{{< /collapse >}}

The chart below shows that the standard deviation of the estimate (of \\(Q(a)\\)) decreases as the number of pulls increases.

{{< chart type="line" palette="math" title="Std of Q(a) estimate vs number of pulls" labels="1, 5, 10, 20, 50" data="1, 0.45, 0.32, 0.22, 0.14" xLabel="Pulls" yLabel="Std" >}}

---

## Professor's hints

- Always use \\(n-1\\) (not \\(n\\)) in the denominator for the *unbiased* sample variance when you are estimating the variance of a distribution from data. Using \\(n\\) gives the MLE of the variance but is biased.
- In RL, “sample” usually means one trajectory or one reward draw. “Sample mean” then means averaging over many such trajectories or draws.
- When you see \\(\mathbb{E}_\pi[...]\\) in RL, it means “expectation under policy \\(\pi\\)” — i.e. over trajectories generated by following \\(\pi\\).

---

## Common pitfalls

- **Confusing expectation with one sample:** The expected reward of an arm is not the same as the reward you got on one pull. Expectation is a property of the distribution; one sample is random.
- **Using \\(n\\) instead of \\(n-1\\) for sample variance:** For small \\(n\\) the difference matters. Stick to \\(n-1\\) when you want an unbiased estimate of the population variance.
- **Assuming independence when it is not:** In RL, consecutive rewards in one episode are often *not* independent. Monte Carlo still works because we average over many *independent episodes*.
