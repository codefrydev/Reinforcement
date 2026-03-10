---
title: "Calculus"
description: "Derivatives, chain rule, sigmoid and softmax — with RL motivation and explained solutions."
date: 2026-03-10T00:00:00Z
draft: false
tags: ["calculus", "derivatives", "chain rule", "sigmoid", "softmax", "preliminary"]
keywords: ["calculus", "derivatives", "chain rule", "sigmoid softmax", "RL motivation"]
---

This page covers the calculus you need for the preliminary assessment: derivatives of common functions, the chain rule, and how they appear in logistic regression and policy gradients. [Back to Preliminary](../).

---

## Why this matters for RL

Policy gradients and loss-based updates use derivatives and the chain rule. You don’t need to derive everything by hand in practice (autograd does it), but you need to understand what a gradient is and how it’s used. The sigmoid and chain rule appear in logistic policies and in backpropagation.

### Learning objectives

Compute derivatives of \\(\ln(1+e^x)\\) and simple composites; state and apply the chain rule; connect these to sigmoid and to RL updates.

---

## Core concepts

- Derivative rules: \\(\frac{d}{dx}e^x = e^x\\), \\(\frac{d}{dx}\ln x = \frac{1}{x}\\). So \\(\frac{d}{dx}\ln(1+e^x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (sigmoid).
- Chain rule: If \\(y = f(u)\\) and \\(u = g(x)\\), then \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\). For compositions, multiply derivatives along the path.
- In RL: Gradient descent minimizes loss: \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). Policy gradient *maximizes* return: \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\). Backprop is the chain rule applied layer by layer.

---

## Worked problems (with explanations)

### 1. Derivative of \\(\ln(1+e^x)\\) (Q5)

Q: Compute the derivative of \\(f(x) = \ln(1+e^x)\\) with respect to \\(x\\). This function appears in logistic regression and softmax.

{{< collapse summary="Answer and explanation" >}}
\\(f'(x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (the sigmoid function).

### Derivation

Let \\(u = 1 + e^x\\). Then \\(f = \ln u\\), so \\(\frac{df}{du} = \frac{1}{u}\\) and \\(\frac{du}{dx} = e^x\\). By the chain rule, \\(\frac{df}{dx} = \frac{1}{u} \cdot e^x = \frac{e^x}{1+e^x}\\).

### Explanation

The sigmoid \\(\sigma(x) = \frac{e^x}{1+e^x}\\) is the derivative of the “softplus” \\(\ln(1+e^x)\\). In logistic regression, the log-likelihood involves \\(\ln(1+e^x)\\); its gradient with respect to the linear predictor involves the sigmoid. In RL, sigmoid is used for binary action probabilities and appears in policy gradient formulas.
{{< /collapse >}}

---

### 2. Chain rule statement and example (Q6)

Q: What is the chain rule for derivatives? Give an example: if \\(y = f(u)\\) and \\(u = g(x)\\), express \\(\frac{dy}{dx}\\). Apply to \\(y = \sin(x^2)\\).

{{< collapse summary="Answer and explanation" >}}
\\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\).

### Example

\\(y = \sin(x^2)\\). Let \\(u = x^2\\), so \\(y = \sin u\\). Then \\(\frac{dy}{du} = \cos u\\) and \\(\frac{du}{dx} = 2x\\). So \\(\frac{dy}{dx} = \cos(x^2) \cdot 2x\\).

### Explanation

The chain rule says: the rate of change of \\(y\\) with respect to \\(x\\) is the rate of change of \\(y\\) with respect to \\(u\\) times the rate of change of \\(u\\) with respect to \\(x\\). In neural networks, the output is a long composition of functions; backprop multiplies derivatives along the path from output to each parameter, which is exactly the chain rule.
{{< /collapse >}}

---

### 3. Another derivative by chain rule

Q: Compute \\(\frac{d}{dx}\bigl(e^{x^2}\bigr)\\).

{{< collapse summary="Answer and explanation" >}}
Let \\(u = x^2\\), so \\(y = e^u\\). Then \\(\frac{dy}{du} = e^u\\) and \\(\frac{du}{dx} = 2x\\). So \\(\frac{dy}{dx} = e^{x^2} \cdot 2x\\).

### Explanation

The inner function is \\(x^2\\), so we must multiply by \\(2x\\). Forgetting the inner derivative is a common mistake. In RL, when we differentiate a loss with respect to an early layer’s weights, we get a product of many such factors (one per layer) via the chain rule.
{{< /collapse >}}

---

## Math example: chain rule tree

For \\(y = f(g(h(x)))\\):
- \\(\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}\\).

Each factor is the derivative of one “layer” with respect to its input. In a neural network, \\(h\\) might be the first layer, \\(g\\) the second, \\(f\\) the loss. Backprop computes these derivatives from output to input and multiplies them. So the chain rule is the reason we can train deep networks: we break the gradient into local pieces and multiply.

---

## Code example (with explanation)

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = (1 + x**2).sqrt()   # sqrt(1 + x^2)
y.backward()
print(x.grad)            # d/dx sqrt(1+x^2) = x/sqrt(1+x^2) -> at x=2: 2/sqrt(5) ≈ 0.894
```

### Explanation

PyTorch builds a computation graph. When we call `y.backward()`, it applies the chain rule: the gradient of \\(\sqrt{u}\\) is \\(1/(2\sqrt{u})\\), and \\(du/dx = 2x\\), so \\(dy/dx = x/\sqrt{1+x^2}\\). At \\(x=2\\), that is \\(2/\sqrt{5}\\). Autograd does this automatically for any composition—that’s how we train policies and value networks in RL.
---

## Professor's hints

- Policy gradient *maximizes* return, so the update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\) (plus sign). Loss minimization uses minus.
- Always include the inner derivative in the chain rule: for \\(f(g(x))\\), multiply by \\(g'(x)\\).
- In RL, we differentiate the log-probability or value prediction with respect to parameters; the reward is usually not differentiated.

---

## Common pitfalls

- Maximize vs minimize: Policy gradient uses a *plus* in the update; loss minimization uses *minus*. Flipping the sign flips the direction of learning.
- Forgetting the inner derivative: \\(\frac{d}{dx}e^{x^2} = e^{x^2} \cdot 2x\\), not just \\(e^{x^2}\\).
- Treating constants as variables: When differentiating w.r.t. \\(\theta\\), quantities that don’t depend on \\(\theta\\) (e.g. observed rewards) have derivative zero.
