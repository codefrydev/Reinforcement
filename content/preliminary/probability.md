---
title: "Probability & Statistics"
description: "Sample mean, variance, expectation, and law of large numbers — with bandit-style problems and explained solutions."
date: 2026-03-10T00:00:00Z
draft: false
difficulty: 3
tags: ["probability", "sample mean", "variance", "expectation", "bandits", "preliminary"]
keywords: ["probability", "sample mean variance", "expectation", "law of large numbers", "bandit"]
weight: 3
roadmap_icon: "calculator"
roadmap_color: "green"
roadmap_phase_label: "Topic 3 · Probability"
---

This page covers the probability and statistics you need for the preliminary assessment: sample mean, unbiased sample variance, expectation vs sample average, and the law of large numbers. [Back to Preliminary](../).

---

## Why this matters for RL

In reinforcement learning, rewards are often random and value functions are *expected* returns. Bandits, Monte Carlo methods, and policy evaluation all rely on expectations and sample averages. You need to compute and interpret sample means and variances by hand and in code.

### Learning objectives

Compute sample mean and unbiased sample variance from data; explain the difference between expectation and sample average; state the law of large numbers; relate these ideas to bandits and Monte Carlo.

---

## Core concepts

- Sample mean: \\(\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i\\). It estimates the expected value from data.
- Unbiased sample variance: \\(\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2\\). The \\(n-1\\) denominator makes it unbiased when estimating the population variance.
- Expectation \\(\mathbb{E}[X]\\) is the theoretical long-run average; the sample average is the empirical estimate. By the law of large numbers, the sample average converges to \\(\mathbb{E}[X]\\) as \\(n \to \infty\\).

---

## Worked problems (with explanations)

### 1. Bandit: sample mean and variance (Q1)

Q: In a multi-armed bandit with 3 arms, the true reward distributions are \\(\mathcal{N}(0,1), \mathcal{N}(1,1), \mathcal{N}(-0.5,1)\\). If you pull arm 2 five times, you observe rewards [1.2, 0.8, 1.5, 0.3, 2.1]. What is the sample mean estimate of arm 2's expected reward? What is the unbiased sample variance?

{{< collapse summary="Answer and explanation" >}}
Sample mean: \\(\bar{x} = (1.2+0.8+1.5+0.3+2.1)/5 = 5.9/5 = 1.18\\).

Sample variance: Use \\(\frac{1}{n-1}\sum (x_i - \bar{x})^2\\). Deviations from 1.18: (1.2−1.18)² + (0.8−1.18)² + (1.5−1.18)² + (0.3−1.18)² + (2.1−1.18)² = 0.0004 + 0.1444 + 0.1024 + 0.7744 + 0.8464 = 1.868. So variance = 1.868/4 = 0.467.

### Explanation

The sample mean 1.18 is our best estimate of arm 2’s true expected reward (which is 1 for \\(\mathcal{N}(1,1)\\)). With only 5 samples we don’t expect it to be exactly 1. The sample variance estimates how spread out the rewards are; we divide by \\(n-1\\) (not \\(n\\)) so that on average this estimate equals the true variance of the distribution. In a bandit, we use these quantities to compare arms and to quantify uncertainty.
{{< /collapse >}}

---

### 2. Expectation vs sample average (Q2)

Q: What is the difference between the expected value of a random variable and a sample average? When do they coincide (in the limit)?

{{< collapse summary="Answer and explanation" >}}
The expected value \\(\mathbb{E}[X]\\) is a theoretical quantity: the long-run average if you could repeat the experiment infinitely many times, weighted by the distribution. The sample average \\(\bar{x} = \frac{1}{n}\sum_i x_i\\) is an empirical quantity: the average of \\(n\\) observations you actually have.

They coincide in the limit by the law of large numbers: as \\(n \to \infty\\), the sample average converges to \\(\mathbb{E}[X]\\) (under mild conditions). So with more and more data, your estimate gets closer to the true expectation.

### Explanation

In RL we never know true expectations (e.g. true \\(V(s)\\) or true arm means). We estimate them from samples. The law of large numbers justifies why Monte Carlo estimation works: average returns over many episodes converge to the expected return.
{{< /collapse >}}

---

### 3. By hand: another bandit-style set

Q: An arm has true expected reward 1. You get 4 samples: 1.2, 0.8, 1.0, 1.4. What is the sample mean? What is the (unbiased) sample variance?

{{< collapse summary="Answer and explanation" >}}
Sample mean: \\(\bar{x} = (1.2+0.8+1.0+1.4)/4 = 4.4/4 = 1.1\\).

Deviations from 1.1: (1.2−1.1)² = 0.01, (0.8−1.1)² = 0.09, (1.0−1.1)² = 0.01, (1.4−1.1)² = 0.09. Sum = 0.20. Unbiased variance = 0.20/(4−1) = 0.067 (approximately).

### Explanation

With \\(n=4\\), we lose one degree of freedom when we use the same data to estimate the mean, so we divide by \\(n-1=3\\) for an unbiased estimate of the variance. The sample mean 1.1 is close to the true mean 1; more pulls would refine the estimate.
{{< /collapse >}}

---

## Math example: law of large numbers

### Setup

Suppose a random variable \\(X\\) has \\(\mathbb{E}[X] = \mu\\) and \\(\mathrm{Var}(X) = \sigma^2\\). Draw \\(n\\) i.i.d. samples \\(x_1,\ldots,x_n\\) and form \\(\bar{x} = \frac{1}{n}\sum_i x_i\\).

### Step 1

\\(\mathbb{E}[\bar{x}] = \frac{1}{n}\sum_i \mathbb{E}[x_i] = \mu\\). So the sample mean is *unbiased*: its expected value is the true mean.

### Step 2

\\(\mathrm{Var}(\bar{x}) = \frac{\sigma^2}{n}\\) (variance of a sum of independent RVs, divided by \\(n^2\\)). So as \\(n\\) grows, the variance of \\(\bar{x}\\) shrinks.

### Step 3

(Law of large numbers) As \\(n \to \infty\\), \\(\bar{x} \to \mu\\) (in probability or almost surely, under standard conditions). So we can estimate \\(\mu\\) by averaging many observations.

### Explanation

This is why bandit algorithms and Monte Carlo methods work: we don’t need the distribution in closed form; we just need many samples and we average them. The more episodes or pulls, the better the estimate.

---

## Code examples (with explanations)

### Sample mean and sample variance in Python

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

### Explanation

`sample_mean` is the usual average. For `sample_variance` we use \\(n-1\\) in the denominator so that when we are estimating the variance of the underlying distribution from a sample, the expected value of this statistic equals the true variance. The list comprehension computes \\((x_i - \bar{x})^2\\) for each observation. In RL you’ll do similar operations on reward batches or returns.

---

## Professor's hints

- Always use \\(n-1\\) (not \\(n\\)) in the denominator for the *unbiased* sample variance when estimating the variance of a distribution from data.
- In RL, “sample” often means one trajectory or one reward draw; “sample mean” means averaging over many such draws or episodes.
- When you see \\(\mathbb{E}_\pi[...]\\) in RL, it means expectation under policy \\(\pi\\) (over trajectories generated by following \\(\pi\\)).

---

## Common pitfalls

- Confusing expectation with one sample: The expected reward of an arm is not the same as the reward you got on one pull. Expectation is a property of the distribution; one sample is random.
- Using \\(n\\) instead of \\(n-1\\) for sample variance: For small \\(n\\) the difference matters. Use \\(n-1\\) when you want an unbiased estimate of the population variance.
- Assuming independence when it is not: In RL, consecutive rewards in one episode are often *not* independent. Monte Carlo still works because we average over many *independent episodes*.
