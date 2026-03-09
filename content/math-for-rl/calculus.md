---
title: "Calculus"
description: "Derivatives, chain rule, and partial derivatives — with RL motivation and practice."
date: 2026-03-10T00:00:00Z
draft: false
---

This page covers the calculus you need for RL: derivatives, the chain rule, and partial derivatives. [Back to Math for RL](/math-for-rl/).

---

## Core concepts

### Derivatives

The **derivative** of \\(f(x)\\) with respect to \\(x\\) is \\(f'(x)\\) or \\(\frac{df}{dx}\\). It gives the rate of change (slope) of \\(f\\) at \\(x\\). Rules you will use:

- \\(\frac{d}{dx} x^n = n x^{n-1}\\)
- \\(\frac{d}{dx} e^x = e^x\\)
- \\(\frac{d}{dx} \ln x = \frac{1}{x}\\)
- \\(\frac{d}{dx} \ln(1 + e^x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (sigmoid)

**In reinforcement learning:** Loss functions and objective functions are differentiated with respect to parameters. Gradient descent uses \\(\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}\\); policy gradient uses \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\). The sigmoid appears in logistic policy parameterizations and in softmax-related derivatives.

### Chain rule

If \\(y = f(u)\\) and \\(u = g(x)\\), then \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\). For multiple compositions, multiply derivatives along the path.

**Example:** \\(y = \sin(x^2)\\) → \\(u = x^2\\), \\(y = \sin(u)\\) → \\(\frac{dy}{dx} = \cos(u) \cdot 2x = \cos(x^2) \cdot 2x\\).

**In reinforcement learning:** Neural networks are compositions of many functions. Backpropagation is the chain rule applied layer by layer. When you call `loss.backward()` in PyTorch, it is computing gradients via the chain rule.

### Partial derivatives and gradients

For a function \\(f(x_1, x_2, \ldots, x_n)\\), the **partial derivative** \\(\frac{\partial f}{\partial x_i}\\) is the derivative with respect to \\(x_i\\) when all other variables are fixed. The **gradient** \\(\nabla f\\) is the vector of all partial derivatives. For \\(f(w) = w^T A w\\) with symmetric \\(A\\), \\(\nabla_w f = 2 A w\\).

**In reinforcement learning:** The policy gradient theorem involves partial derivatives of the log-probability of the action with respect to each parameter. Loss functions depend on many parameters; we need all partial derivatives to update them.

---

## Practice questions

1. **Derivative:** Compute \\(\frac{d}{dx}\bigl(\ln(1+e^x)\bigr)\\). Show it equals the sigmoid \\(\sigma(x) = \frac{e^x}{1+e^x}\\).
2. **Chain rule:** If \\(y = f(u)\\) and \\(u = g(x)\\), write \\(\frac{dy}{dx}\\) in terms of \\(\frac{dy}{du}\\) and \\(\frac{du}{dx}\\). Apply it to \\(y = (1 + x^2)^{1/2}\\).
3. **Partial:** For \\(f(w_1, w_2) = w_1^2 + w_1 w_2 + w_2^2\\), compute \\(\frac{\partial f}{\partial w_1}\\) and \\(\frac{\partial f}{\partial w_2}\\). Write the gradient \\(\nabla f\\).
4. **RL:** In supervised learning we minimize \\(L(\theta)\\) with \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). In policy gradient we *maximize* expected return \\(J(\theta)\\). Write the analogous parameter update for policy gradient.
5. **Python/PyTorch:** Create a scalar tensor \\(x = 2.0\\) with `requires_grad=True`, compute \\(y = x^2\\), call `y.backward()`, and print `x.grad`. Confirm it equals \\(\frac{dy}{dx} = 2x\\) at \\(x=2\\).

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
