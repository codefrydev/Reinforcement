---
title: "Calculus"
description: "Derivatives, chain rule, and partial derivatives — with RL motivation and practice."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["calculus", "derivatives", "chain rule", "gradients", "math for RL"]
keywords: ["calculus for RL", "derivatives", "chain rule", "partial derivatives", "gradient descent"]
---

This page covers the calculus you need for RL: derivatives, the chain rule, and partial derivatives. [Back to Math for RL](../).

---

## Core concepts

### Derivatives

The **derivative** of \\(f(x)\\) with respect to \\(x\\) is \\(f'(x)\\) or \\(\frac{df}{dx}\\). It gives the rate of change (slope) of \\(f\\) at \\(x\\). Rules you will use:

- \\(\frac{d}{dx} x^n = n x^{n-1}\\)
- \\(\frac{d}{dx} e^x = e^x\\)
- \\(\frac{d}{dx} \ln x = \frac{1}{x}\\)
- \\(\frac{d}{dx} \ln(1 + e^x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (sigmoid)

The chart below shows the sigmoid \\(\sigma(x) = \frac{e^x}{1+e^x}\\): the S-shaped function whose derivative we use in policy parameterizations and softplus.

{{< chart type="line" title="Sigmoid σ(x)" labels="-3, -2, -1, 0, 1, 2, 3" data="0.05, 0.12, 0.27, 0.5, 0.73, 0.88, 0.95" >}}

**In reinforcement learning:** Loss functions and objective functions are differentiated with respect to parameters. Gradient descent uses \\(\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}\\); policy gradient uses \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\). The sigmoid appears in logistic policy parameterizations and in softmax-related derivatives.

### Chain rule

If \\(y = f(u)\\) and \\(u = g(x)\\), then \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\). For multiple compositions, multiply derivatives along the path.

**Example:** \\(y = \sin(x^2)\\) → \\(u = x^2\\), \\(y = \sin(u)\\) → \\(\frac{dy}{dx} = \cos(u) \cdot 2x = \cos(x^2) \cdot 2x\\). The graph below shows \\(y = \sin(x^2)\\) over a few points so you can see the curve whose slope (derivative) we computed.

{{< chart type="line" title="y = sin(x²)" labels="0, 0.5, 1, 1.5, 2, 2.5" data="0, 0.25, 0.84, 0.78, -0.76, 0.07" >}}

**In reinforcement learning:** Neural networks are compositions of many functions. Backpropagation is the chain rule applied layer by layer. When you call `loss.backward()` in PyTorch, it is computing gradients via the chain rule.

### Partial derivatives and gradients

For a function \\(f(x_1, x_2, \ldots, x_n)\\), the **partial derivative** \\(\frac{\partial f}{\partial x_i}\\) is the derivative with respect to \\(x_i\\) when all other variables are fixed. The **gradient** \\(\nabla f\\) is the vector of all partial derivatives. For \\(f(w) = w^T A w\\) with symmetric \\(A\\), \\(\nabla_w f = 2 A w\\).

**In reinforcement learning:** The policy gradient theorem involves partial derivatives of the log-probability of the action with respect to each parameter. Loss functions depend on many parameters; we need all partial derivatives to update them.

---

## Practice questions

1. **Derivative:** Compute \\(\frac{d}{dx}\bigl(\ln(1+e^x)\bigr)\\). Show it equals the sigmoid \\(\sigma(x) = \frac{e^x}{1+e^x}\\).

{{< collapse summary="Answer and explanation" >}}
**Step 1 — Substitute:** Let \\(u = 1 + e^x\\). Then \\(f = \ln u\\), so \\(\frac{df}{du} = \frac{1}{u}\\) and \\(\frac{du}{dx} = e^x\\).

**Step 2 — Chain rule:** \\(\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} = \frac{1}{u} \cdot e^x = \frac{e^x}{1+e^x} = \sigma(x)\\).

**Answer:** \\(\frac{d}{dx}\ln(1+e^x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (the sigmoid).

**Explanation:** The sigmoid is the derivative of the softplus \\(\ln(1+e^x)\\). In logistic regression and policy parameterizations, the gradient of the log-likelihood involves this derivative.
{{< /collapse >}}

---

2. **Chain rule:** If \\(y = f(u)\\) and \\(u = g(x)\\), write \\(\frac{dy}{dx}\\) in terms of \\(\frac{dy}{du}\\) and \\(\frac{du}{dx}\\). Apply it to \\(y = (1 + x^2)^{1/2}\\).

{{< collapse summary="Answer and explanation" >}}
**Statement:** \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\).

**Application:** \\(y = (1+x^2)^{1/2}\\). Let \\(u = 1 + x^2\\), so \\(y = u^{1/2}\\). Then \\(\frac{dy}{du} = \frac{1}{2}u^{-1/2} = \frac{1}{2}(1+x^2)^{-1/2}\\) and \\(\frac{du}{dx} = 2x\\). So \\(\frac{dy}{dx} = \frac{1}{2}(1+x^2)^{-1/2} \cdot 2x = \frac{x}{(1+x^2)^{1/2}} = \frac{x}{\sqrt{1+x^2}}\\).

**Explanation:** The chain rule multiplies derivatives along the path. In neural networks, backprop does this layer by layer; autograd libraries compute it automatically.
{{< /collapse >}}

---

3. **Partial:** For \\(f(w_1, w_2) = w_1^2 + w_1 w_2 + w_2^2\\), compute \\(\frac{\partial f}{\partial w_1}\\) and \\(\frac{\partial f}{\partial w_2}\\). Write the gradient \\(\nabla f\\).

{{< collapse summary="Answer and explanation" >}}
**Step 1 — \\(\frac{\partial f}{\partial w_1}\\):** Treat \\(w_2\\) as constant. \\(\frac{\partial}{\partial w_1}(w_1^2) = 2w_1\\), \\(\frac{\partial}{\partial w_1}(w_1 w_2) = w_2\\), \\(\frac{\partial}{\partial w_1}(w_2^2) = 0\\). So \\(\frac{\partial f}{\partial w_1} = 2w_1 + w_2\\).

**Step 2 — \\(\frac{\partial f}{\partial w_2}\\):** Similarly \\(\frac{\partial f}{\partial w_2} = w_1 + 2w_2\\).

**Gradient:** \\(\nabla f = \bigl[2w_1 + w_2,\; w_1 + 2w_2\bigr]^T\\).

**Explanation:** The gradient is the vector of partial derivatives. In RL we need all partials to update each parameter (e.g. in policy or value networks).
{{< /collapse >}}

---

4. **RL:** In supervised learning we minimize \\(L(\theta)\\) with \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). In policy gradient we *maximize* expected return \\(J(\theta)\\). Write the analogous parameter update for policy gradient.

{{< collapse summary="Answer and explanation" >}}
We *maximize* \\(J(\theta)\\), so we move in the direction of the gradient (gradient *ascent*). The update is:

\\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\).

**Explanation:** Minimization uses minus (gradient descent); maximization uses plus. Policy gradient increases return by taking a step in the direction that increases \\(J\\).
{{< /collapse >}}

---

5. **Python/PyTorch:** Create a scalar tensor \\(x = 2.0\\) with `requires_grad=True`, compute \\(y = x^2\\), call `y.backward()`, and print `x.grad`. Confirm it equals \\(\frac{dy}{dx} = 2x\\) at \\(x=2\\).

{{< collapse summary="Answer and explanation" >}}
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(4.)
```

**Check:** \\(\frac{dy}{dx} = 2x\\). At \\(x = 2\\), \\(\frac{dy}{dx} = 4\\). So `x.grad` should be 4.0. PyTorch’s autograd applies the chain rule; for \\(y = x^2\\) it stores \\(2x\\) in `x.grad` when we call `backward()`.

**Explanation:** This is how we get gradients for policy and value parameters in RL—PyTorch builds the computation graph and computes derivatives automatically.
{{< /collapse >}}

---

6. **By hand:** For \\(f(x) = x^2 e^x\\), compute \\(f'(x)\\) using the product rule. Evaluate at \\(x = 0\\).

{{< collapse summary="Answer and explanation" >}}
**Product rule:** \\((uv)' = u'v + uv'\\). With \\(u = x^2\\) and \\(v = e^x\\), we have \\(u' = 2x\\) and \\(v' = e^x\\). So \\(f'(x) = 2x \cdot e^x + x^2 \cdot e^x = e^x(2x + x^2)\\).

**At \\(x = 0\\):** \\(f'(0) = e^0(0 + 0) = 0\\).

**Explanation:** The product rule is used whenever the loss or objective is a product of terms (e.g. policy probability times advantage). At \\(x=0\\) the slope is 0.
{{< /collapse >}}

---

7. **RL:** In **policy gradients**, we often compute \\(\nabla_\theta \log \pi(a|s;\theta)\\). Why is the *log* used instead of \\(\nabla_\theta \pi(a|s)\\)? (Hint: log-derivative trick; \\(\nabla \pi = \pi \nabla \log \pi\\), which keeps the gradient scaled by the probability.)

{{< collapse summary="Answer and explanation" >}}
**Log-derivative trick:** \\(\nabla_\theta \pi(a|s;\theta) = \pi(a|s;\theta) \cdot \nabla_\theta \log \pi(a|s;\theta)\\). So the gradient of the probability is the probability times the gradient of the log-probability.

**Why use log:** (1) The policy gradient theorem involves \\(\nabla_\theta \log \pi\\) weighted by the advantage; the log form keeps the gradient scaled by the probability and leads to a simple unbiased estimator. (2) In practice, \\(\log \pi\\) is numerically stable (we often work with log-probabilities to avoid underflow). (3) The resulting gradient update has a clean form (e.g. REINFORCE: increase log-probability of actions that got high return).

**Explanation:** So we use \\(\nabla_\theta \log \pi\\) in the update; the “trick” shows this is equivalent to a probability-weighted gradient of \\(\pi\\), which is what the policy gradient theorem requires.
{{< /collapse >}}

---

## Professor's hints

- The chain rule is why we can train deep networks: we break the gradient into local pieces (each layer’s derivative) and multiply them. Autograd libraries do this automatically.
- In RL, “gradient” almost always means “with respect to the policy or value function parameters.” The reward is usually not differentiated; we differentiate the log-probability or the value prediction.
- When you see \\(\nabla_\theta \log \pi(a|s;\theta)\\), that is the gradient of the log-probability of the action under the policy—this vector appears in the policy gradient theorem.

---

## Common pitfalls

- **Maximize vs minimize:** Policy gradient *maximizes* return, so the update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\) (plus, not minus). Loss minimization uses minus. Mixing them up flips the direction of learning.
- **Forgetting the inner derivative in the chain rule:** For \\(f(g(x))\\), you must multiply by \\(g'(x)\\). For example, \\(\frac{d}{dx}e^{x^2} = e^{x^2} \cdot 2x\\).
- **Treating constants as variables:** When you differentiate with respect to \\(\theta\\), quantities that do not depend on \\(\theta\\) (e.g. rewards already observed) are constants; their derivative is zero. Only the part that depends on \\(\theta\\) (e.g. \\(\log \pi(a|s;\theta)\\)) contributes to the gradient.
