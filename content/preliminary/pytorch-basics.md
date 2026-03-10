---
title: "PyTorch Basics"
description: "Tensors, requires_grad, backward, and autograd — with RL-relevant examples and explanations."
date: 2026-03-10T00:00:00Z
draft: false
---

This page covers the PyTorch you need for the preliminary assessment: creating tensors, enabling gradients, and computing gradients with `backward()`. [Back to Preliminary](../).

---

## Why this matters for RL

States, actions, and batches are tensors; policy and value networks use `nn.Module` and autograd. Policy gradient and value losses require gradients; `backward()` and `.grad` are central. You need to create tensors with `requires_grad=True` and interpret the gradients PyTorch computes.

### Learning objectives

Create a tensor with `requires_grad=True`; compute a scalar function of it and call `backward()`; read the gradient from `.grad`. Relate this to loss minimization and policy gradient.

---

## Core concepts

- Tensors: `torch.tensor(value, requires_grad=True)` for scalars; `torch.zeros(n, m)` for arrays. Operations build a computation graph when inputs require gradients.
- Autograd: Call `.backward()` on a scalar tensor to compute gradients of that scalar with respect to all tensors that have `requires_grad=True` and participated in the computation. Gradients accumulate in `.grad`.
- Gradient: For \\(y = f(x)\\), after `y.backward()`, `x.grad` holds \\(dy/dx\\) (or the gradient vector for multi-element \\(x\\)).

---

## Worked problems (with explanations)

### 1. Gradient of \\(y = x^2\\) at \\(x = 2\\) (Q9)

Q: In PyTorch, how do you create a tensor requiring gradient, and how do you compute the gradient of \\(y = x^2\\) with respect to \\(x\\) for \\(x=2.0\\)?

{{< collapse summary="Answer and explanation" >}}
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # tensor(4.)
```

### Explanation

We create a scalar tensor \\(x = 2\\) with `requires_grad=True` so PyTorch will track operations. Then \\(y = x^2\\) is recorded in the computation graph. `y.backward()` computes \\(dy/dx = 2x\\) and stores it in `x.grad`. At \\(x=2\\), \\(2x=4\\). In RL, the “scalar” is usually a loss (e.g. TD error squared or policy gradient objective), and we call `loss.backward()` to get gradients for all parameters that contributed to the loss.
{{< /collapse >}}

---

### 2. Gradient of \\(y = x^T x\\) (vector)

Q: Create a 3-vector \\(x\\) with `requires_grad=True`, set \\(y = x^T x\\) (sum of squares), call `y.backward()`, and interpret `x.grad`.

{{< collapse summary="Answer and explanation" >}}
```python
import torch
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x * x).sum()   # or torch.dot(x, x)
y.backward()
print(x.grad)   # tensor([2., 4., 6.])
```

### Explanation

\\(y = x_1^2 + x_2^2 + x_3^2\\), so \\(\frac{\partial y}{\partial x_i} = 2x_i\\). So the gradient is \\(2x = [2, 4, 6]\\). PyTorch stores this in `x.grad`. For a general scalar loss \\(L(x)\\), `x.grad` is \\(\nabla_x L\\). In RL, when \\(x\\) is a parameter vector, we use this gradient in an update like \\(x \leftarrow x - \alpha\, x.grad\\) for minimization.
{{< /collapse >}}

---

### 3. Multiple tensors and clearing gradients

Q: Compute \\(z = x^2 + y^2\\) with \\(x=1.0\\) and \\(y=2.0\\) (both require_grad). After `z.backward()`, what are `x.grad` and `y.grad`? Why might we need to zero gradients before the next backward?

{{< collapse summary="Answer and explanation" >}}
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2 + y**2
z.backward()
print(x.grad, y.grad)  # tensor(2.) tensor(4.)
```

\\(\partial z/\partial x = 2x = 2\\), \\(\partial z/\partial y = 2y = 4\\). So `x.grad=2`, `y.grad=4`.

### Explanation

PyTorch *accumulates* gradients by default (adds into `.grad`). So if you call `backward()` again without zeroing, the new gradients are added to the old ones. In training loops we typically call `optimizer.zero_grad()` before each forward/backward so each step uses only the gradients from the current batch. For a single standalone computation like this, accumulation doesn’t matter.
{{< /collapse >}}

---

## Math example: why \\(dy/dx = 2x\\) for \\(y = x^2\\)

By the power rule, \\(\frac{d}{dx}x^2 = 2x\\). So at any point \\(x\\), the slope of \\(y = x^2\\) is \\(2x\\). PyTorch’s autograd derives this symbolically (or via automatic differentiation) and evaluates it at the current value of \\(x\\). For more complex expressions, it applies the chain rule step by step through the computation graph. That’s exactly what we need for policy gradients: the loss may be a complicated function of many parameters, and we get the gradient of the loss with respect to every parameter in one `backward()` call.

---

## Code example (with explanation)

```python
import torch
# Simulate a small “loss” that depends on parameters
theta = torch.tensor([0.5, -0.5], requires_grad=True)
pred = theta[0] * 2 + theta[1] * 1   # simple linear combo
target = torch.tensor(1.0)
loss = (pred - target) ** 2
loss.backward()
print(theta.grad)   # gradient of (pred - target)^2 w.r.t. theta
```

### Explanation

We have a 2D parameter `theta`, a scalar prediction `pred`, and a squared error loss. After `loss.backward()`, `theta.grad` is the gradient of the loss with respect to `theta`. This is the same pattern used in value function fitting: prediction from parameters, loss from prediction and target, then backward to get parameter updates. In practice we’d use `nn.Parameter` and an optimizer, but the idea is the same.
---

## Professor's hints

- Only scalar tensors can be used as the root for `.backward()` (or pass a gradient tensor for higher-dimensional outputs). For a vector loss, sum it to a scalar first (e.g. `loss.sum().backward()`).
- `requires_grad=True` is needed on any tensor whose gradient you want. Parameters of `nn.Module` are typically `nn.Parameter` (which has `requires_grad=True` by default).
- After `backward()`, the graph is freed by default. So you can’t call `backward()` twice on the same graph unless you use `retain_graph=True` (rare in normal training).

---

## Common pitfalls

- Forgetting `requires_grad=True`: If you create a tensor without it, operations won’t build a graph and `.grad` will stay `None`.
- Calling `backward()` on non-scalar: You must either reduce to a scalar (e.g. `.sum()`) or pass a gradient tensor to `backward(gradient=...)`.
- Not zeroing gradients: In a training loop, zero gradients before each backward (e.g. `optimizer.zero_grad()`) so you don’t accumulate gradients from previous steps.
