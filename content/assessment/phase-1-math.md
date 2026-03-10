---
title: "Phase 1 Self-Check: Math for RL"
description: "10 questions to check readiness after the Math for RL track. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
---

Use this self-check after completing [Probability](../math-for-rl/probability/), [Linear algebra](../math-for-rl/linear-algebra/), and [Calculus](../math-for-rl/calculus/). If you can answer at least 8 correctly and feel comfortable with the concepts, you are ready for Phase 2 and the curriculum.

---

### 1. Probability

**Q:** In a bandit, you pull arm 2 five times and get rewards [0.5, 1.2, 0.8, 1.0, 0.9]. What is the sample mean? What is the unbiased sample variance (use \\(n-1\\) in the denominator)?

{{< collapse summary="Answer" >}}
Sample mean = (0.5+1.2+0.8+1.0+0.9)/5 = **0.88**.  
Deviations from mean: -0.38, 0.32, -0.08, 0.12, 0.02. Squared: 0.1444, 0.1024, 0.0064, 0.0144, 0.0004. Sum = 0.268; variance = 0.268/4 = **0.067**.
{{< /collapse >}}

---

### 2. Probability

**Q:** In one sentence, what does the law of large numbers say about the sample average and the expected value?

{{< collapse summary="Answer" >}}
As the number of samples grows, the sample average converges to the expected value (under mild conditions).
{{< /collapse >}}

---

### 3. Probability & RL

**Q:** Why do we need many episodes in Monte Carlo prediction to get a good estimate of \\(V(s)\\)? Relate to the law of large numbers.

{{< collapse summary="Answer" >}}
\\(V(s)\\) is the *expected* return from \\(s\\); we estimate it by averaging returns from many episodes that visit \\(s\\). The law of large numbers says this sample average converges to the expectation as the number of episodes (samples) increases.
{{< /collapse >}}

---

### 4. Linear algebra

**Q:** Given \\(x = [1, 0, 2]^T\\) and \\(y = [3, 1, 1]^T\\), compute the dot product \\(x^T y\\).

{{< collapse summary="Answer" >}}
\\(1\\cdot 3 + 0\\cdot 1 + 2\\cdot 1 = 3 + 0 + 2 = **5\\).
{{< /collapse >}}

---

### 5. Linear algebra

**Q:** If \\(f(w) = a^T w\\) with \\(a\\) constant, what is \\(\nabla_w f\\)?

{{< collapse summary="Answer" >}}
\\(\nabla_w f = a\\) (the gradient of a linear function is the coefficient vector).
{{< /collapse >}}

---

### 6. Linear algebra & RL

**Q:** In linear value approximation \\(V(s) = w^T \\phi(s)\\), what role does \\(w\\) play? What is \\(\phi(s)\\)?

{{< collapse summary="Answer" >}}
\\(w\\) is the weight vector we learn; \\(\phi(s)\\) is the feature vector for state \\(s\\) (e.g. tile coding or hand-designed features). The dot product \\(w^T \\phi(s)\\) gives the predicted value.
{{< /collapse >}}

---

### 7. Calculus

**Q:** Compute \\(\frac{d}{dx} \\ln(1 + e^x)\\). What well-known function is this?

{{< collapse summary="Answer" >}}
\\(\frac{e^x}{1+e^x} = \\sigma(x)\\), the **sigmoid** function.
{{< /collapse >}}

---

### 8. Calculus

**Q:** State the chain rule: if \\(y = f(u)\\) and \\(u = g(x)\\), write \\(\frac{dy}{dx}\\) in terms of \\(\frac{dy}{du}\\) and \\(\frac{du}{dx}\\).

{{< collapse summary="Answer" >}}
\\(\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}\\).
{{< /collapse >}}

---

### 9. Calculus & RL

**Q:** In policy gradient we *maximize* expected return \\(J(\\theta)\\). Write the parameter update (gradient ascent) with step size \\(\\alpha\\).

{{< collapse summary="Answer" >}}
\\(\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)\\). (We use plus because we maximize; in loss minimization we use minus.)
{{< /collapse >}}

---

### 10. Mixed

**Q:** For a discrete random variable \\(X\\) with outcomes \\(x_i\\) and probabilities \\(p_i\\), write the formula for \\(\\mathbb{E}[X]\\).

{{< collapse summary="Answer" >}}
\\(\\mathbb{E}[X] = \\sum_i x_i p_i\\).
{{< /collapse >}}

---

**Next step:** If you passed the self-check, go to [Phase 2 — Prerequisites](../learning-path/#phase-2--prerequisites-tools-and-libraries) and the [Python prerequisite](../prerequisites/python/). Then continue with the [Learning path](../learning-path/) and [Volume 1](../curriculum/volume-01/).
