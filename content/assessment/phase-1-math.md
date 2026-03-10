---
title: "Phase 1 Self-Check: Math for RL"
description: "10 questions to check readiness after the Math for RL track. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["assessment", "phase 1", "math", "self-check", "solutions"]
keywords: ["phase 1 assessment", "math for RL", "readiness", "10 questions", "solutions", "probability linear algebra calculus"]
---

Use this self-check after completing [Probability](../math-for-rl/probability/), [Linear algebra](../math-for-rl/linear-algebra/), and [Calculus](../math-for-rl/calculus/). If you can answer at least 8 correctly and feel comfortable with the concepts, you are ready for Phase 2 and the curriculum.

---

### 1. Probability

**Q:** In a bandit, you pull arm 2 five times and get rewards [0.5, 1.2, 0.8, 1.0, 0.9]. What is the sample mean? What is the unbiased sample variance (use \\(n-1\\) in the denominator)?

{{< collapse summary="Answer" >}}
**Step 1 — Sample mean:** Sum = 0.5 + 1.2 + 0.8 + 1.0 + 0.9 = 4.4; mean = 4.4/5 = **0.88**.

**Step 2 — Deviations from mean:** -0.38, 0.32, -0.08, 0.12, 0.02. Squared: 0.1444, 0.1024, 0.0064, 0.0144, 0.0004. Sum = 0.268.

**Step 3 — Unbiased variance:** Divide by \\(n-1 = 4\\): 0.268/4 = **0.067**.

We use \\(n-1\\) so that on average the sample variance equals the true variance of the distribution (unbiased estimate). In bandits we use this to compare arms and measure uncertainty.
{{< /collapse >}}

---

### 2. Probability

**Q:** In one sentence, what does the law of large numbers say about the sample average and the expected value?

{{< collapse summary="Answer" >}}
As the number of samples grows, the sample average converges to the expected value (under mild conditions).

**Why it matters in RL:** We never know true expectations (e.g. true \\(V(s)\\) or arm means); we estimate them by averaging many returns or rewards, and the law of large numbers justifies why that works.
{{< /collapse >}}

---

### 3. Probability & RL

**Q:** Why do we need many episodes in Monte Carlo prediction to get a good estimate of \\(V(s)\\)? Relate to the law of large numbers.

{{< collapse summary="Answer" >}}
\\(V(s)\\) is the *expected* return from \\(s\\); we don’t have the distribution, only samples (returns from episodes that visit \\(s\\)). We estimate \\(V(s)\\) by the sample average of those returns. The law of large numbers says this average converges to the expectation as the number of episodes increases—so we need many episodes for a good estimate.

**In RL:** With few episodes the estimate is noisy; with many it stabilizes, just like estimating a bandit arm’s mean by averaging many pulls.
{{< /collapse >}}

---

### 4. Linear algebra

**Q:** Given \\(x = [1, 0, 2]^T\\) and \\(y = [3, 1, 1]^T\\), compute the dot product \\(x^T y\\).

{{< collapse summary="Answer" >}}
**Step 1:** \\(x^T y = x_1 y_1 + x_2 y_2 + x_3 y_3 = 1\cdot 3 + 0\cdot 1 + 2\cdot 1 = 3 + 0 + 2 = **5**\\).

The dot product is the sum of products of corresponding components. In RL, \\(V(s) = w^T \phi(s)\\) is a dot product between weights and features.
{{< /collapse >}}

---

### 5. Linear algebra

**Q:** If \\(f(w) = a^T w\\) with \\(a\\) constant, what is \\(\nabla_w f\\)?

{{< collapse summary="Answer" >}}
\\(\nabla_w f = a\\).

**Why:** For \\(f(w) = a^T w = a_1 w_1 + \cdots + a_n w_n\\), we have \\(\frac{\partial f}{\partial w_i} = a_i\\) for each \\(i\\), so the gradient (vector of partials) is \\(a\\). In RL, linear value functions and many loss terms have this form.
{{< /collapse >}}

---

### 6. Linear algebra & RL

**Q:** In linear value approximation \\(V(s) = w^T \\phi(s)\\), what role does \\(w\\) play? What is \\(\phi(s)\\)?

{{< collapse summary="Answer" >}}
\\(w\\) is the **weight vector** we learn (the parameters of the linear approximator). \\(\phi(s)\\) is the **feature vector** for state \\(s\\) (e.g. tile coding, hand-designed features, or the raw state if identity). The dot product \\(w^T \phi(s)\\) gives the predicted value.

**In RL:** We update \\(w\\) from experience so that \\(w^T \phi(s)\\) approximates the true value or return; the features summarize the state in a form suitable for linear combination.
{{< /collapse >}}

---

### 7. Calculus

**Q:** Compute \\(\frac{d}{dx} \\ln(1 + e^x)\\). What well-known function is this?

{{< collapse summary="Answer" >}}
**Step 1:** Let \\(u = 1 + e^x\\). Then \\(f = \ln u\\), so \\(\frac{df}{du} = \frac{1}{u}\\) and \\(\frac{du}{dx} = e^x\\). **Step 2:** By the chain rule, \\(\frac{df}{dx} = \frac{1}{u} \cdot e^x = \frac{e^x}{1+e^x} = \sigma(x)\\), the **sigmoid**.

The sigmoid is the derivative of the softplus \\(\ln(1+e^x)\\); it appears in logistic policies and in gradient formulas for binary action probabilities.
{{< /collapse >}}

---

### 8. Calculus

**Q:** State the chain rule: if \\(y = f(u)\\) and \\(u = g(x)\\), write \\(\frac{dy}{dx}\\) in terms of \\(\frac{dy}{du}\\) and \\(\frac{du}{dx}\\).

{{< collapse summary="Answer" >}}
\\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\).

**In RL:** Backpropagation is the chain rule applied layer by layer; when you call `loss.backward()` in PyTorch, it multiplies derivatives along the path from output to each parameter.
{{< /collapse >}}

---

### 9. Calculus & RL

**Q:** In policy gradient we *maximize* expected return \\(J(\\theta)\\). Write the parameter update (gradient ascent) with step size \\(\\alpha\\).

{{< collapse summary="Answer" >}}
\\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\).

We use **plus** because we *maximize* return (gradient ascent). In supervised learning we *minimize* loss, so the update uses **minus** (gradient descent). Flipping the sign flips the direction of learning.
{{< /collapse >}}

---

### 10. Mixed

**Q:** For a discrete random variable \\(X\\) with outcomes \\(x_i\\) and probabilities \\(p_i\\), write the formula for \\(\\mathbb{E}[X]\\).

{{< collapse summary="Answer" >}}
\\(\mathbb{E}[X] = \sum_i x_i p_i\\), where \\(x_i\\) are the outcomes and \\(p_i\\) their probabilities.

**In RL:** Value functions are expectations of return; bandit arm means are expectations of reward. We estimate these from samples because we usually don’t know the full distribution.
{{< /collapse >}}

---

**Next step:** If you passed the self-check, go to [Phase 2 — Prerequisites](../learning-path/#phase-2--prerequisites-tools-and-libraries) and the [Python prerequisite](../prerequisites/python/). Then continue with the [Learning path](../learning-path/) and [Volume 1](../curriculum/volume-01/).
