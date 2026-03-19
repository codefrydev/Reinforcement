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

{{< chart type="line" palette="math" title="Sigmoid σ(x)" labels="-3, -2, -1, 0, 1, 2, 3" data="0.05, 0.12, 0.27, 0.5, 0.73, 0.88, 0.95" xLabel="x" yLabel="σ(x)" >}}

**In reinforcement learning:** Loss functions and objective functions are differentiated with respect to parameters. Gradient descent uses \\(\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}\\); policy gradient uses \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\). The sigmoid appears in logistic policy parameterizations and in softmax-related derivatives.

### Chain rule

If \\(y = f(u)\\) and \\(u = g(x)\\), then \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\). For multiple compositions, multiply derivatives along the path.

**Example:** \\(y = \sin(x^2)\\) → \\(u = x^2\\), \\(y = \sin(u)\\) → \\(\frac{dy}{dx} = \cos(u) \cdot 2x = \cos(x^2) \cdot 2x\\). The graph below shows \\(y = \sin(x^2)\\) over a few points so you can see the curve whose slope (derivative) we computed.

{{< chart type="line" palette="math" title="y = sin(x²)" labels="0, 0.5, 1, 1.5, 2, 2.5" data="0, 0.25, 0.84, 0.78, -0.76, -0.03" xLabel="x" yLabel="y" >}}

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

**Python:** `import numpy as np; x = np.linspace(-3,3,50); softplus = np.log1p(np.exp(x)); sigmoid = np.exp(x)/(1+np.exp(x)); (np.gradient(softplus, x) - sigmoid).max()` ≈ 0 (numerically).
{{< /collapse >}}

The derivative \\(\frac{d}{dx}\ln(1+e^x) = \sigma(x)\\). The chart below shows the sigmoid (same as in Core concepts).

{{< chart type="line" palette="math" title="σ(x) = derivative of softplus" labels="-3, -2, -1, 0, 1, 2, 3" data="0.05, 0.12, 0.27, 0.5, 0.73, 0.88, 0.95" xLabel="x" yLabel="σ(x)" >}}

---

2. **Chain rule:** If \\(y = f(u)\\) and \\(u = g(x)\\), write \\(\frac{dy}{dx}\\) in terms of \\(\frac{dy}{du}\\) and \\(\frac{du}{dx}\\). Apply it to \\(y = (1 + x^2)^{1/2}\\).

{{< collapse summary="Answer and explanation" >}}
**Statement:** \\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\).

**Application:** \\(y = (1+x^2)^{1/2}\\). Let \\(u = 1 + x^2\\), so \\(y = u^{1/2}\\). Then \\(\frac{dy}{du} = \frac{1}{2}u^{-1/2} = \frac{1}{2}(1+x^2)^{-1/2}\\) and \\(\frac{du}{dx} = 2x\\). So \\(\frac{dy}{dx} = \frac{1}{2}(1+x^2)^{-1/2} \cdot 2x = \frac{x}{(1+x^2)^{1/2}} = \frac{x}{\sqrt{1+x^2}}\\).

**Explanation:** The chain rule multiplies derivatives along the path. In neural networks, backprop does this layer by layer; autograd libraries compute it automatically.

**Python:** `import numpy as np; x = 0.5; u = 1+x**2; dy_du = 0.5*u**(-0.5); du_dx = 2*x; print(dy_du * du_dx)` → slope at x=0.5. Or use a numerical derivative: `h = 1e-5; (np.sqrt(1+(x+h)**2) - np.sqrt(1+(x-h)**2)) / (2*h)`.
{{< /collapse >}}

For \\(y = \sqrt{1+x^2}\\), the slope \\(dy/dx = x/\sqrt{1+x^2}\\) varies with \\(x\\). The chart below shows \\(y\\) at a few points.

{{< chart type="line" palette="math" title="y = √(1+x²) at sample points" labels="-2, -1, 0, 1, 2" data="2.24, 1.41, 1, 1.41, 2.24" xLabel="x" yLabel="y" >}}

---

3. **Partial:** For \\(f(w_1, w_2) = w_1^2 + w_1 w_2 + w_2^2\\), compute \\(\frac{\partial f}{\partial w_1}\\) and \\(\frac{\partial f}{\partial w_2}\\). Write the gradient \\(\nabla f\\).

{{< collapse summary="Answer and explanation" >}}
**Step 1 — \\(\frac{\partial f}{\partial w_1}\\):** Treat \\(w_2\\) as constant. \\(\frac{\partial}{\partial w_1}(w_1^2) = 2w_1\\), \\(\frac{\partial}{\partial w_1}(w_1 w_2) = w_2\\), \\(\frac{\partial}{\partial w_1}(w_2^2) = 0\\). So \\(\frac{\partial f}{\partial w_1} = 2w_1 + w_2\\).

**Step 2 — \\(\frac{\partial f}{\partial w_2}\\):** Similarly \\(\frac{\partial f}{\partial w_2} = w_1 + 2w_2\\).

**Gradient:** \\(\nabla f = \bigl[2w_1 + w_2,\; w_1 + 2w_2\bigr]^T\\).

**Explanation:** The gradient is the vector of partial derivatives. In RL we need all partials to update each parameter (e.g. in policy or value networks).

**Python:** `import numpy as np; f = lambda w1,w2: w1**2 + w1*w2 + w2**2; w1,w2 = 1.,2.; grad = np.array([2*w1+w2, w1+2*w2]); print(grad)` → `[4. 5.]`.
{{< /collapse >}}

At \\(w_1=1, w_2=2\\) the gradient is \\([4, 5]^T\\). The chart below shows the two partial derivatives.

{{< chart type="bar" palette="math" title="∇f at (1,2): [4, 5]ᵀ" labels="∂f/∂w₁, ∂f/∂w₂" data="4, 5" yLabel="∂f/∂wᵢ" >}}

---

4. **RL:** In supervised learning we minimize \\(L(\theta)\\) with \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). In policy gradient we *maximize* expected return \\(J(\theta)\\). Write the analogous parameter update for policy gradient.

{{< collapse summary="Answer and explanation" >}}
We *maximize* \\(J(\theta)\\), so we move in the direction of the gradient (gradient *ascent*). The update is:

\\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\).

**Explanation:** Minimization uses minus (gradient descent); maximization uses plus. Policy gradient increases return by taking a step in the direction that increases \\(J\\).

**Python:** `theta = theta + alpha * grad_J` (gradient ascent). Contrast with `theta = theta - alpha * grad_L` for loss minimization.
{{< /collapse >}}

Policy gradient uses gradient *ascent*: \\(J\\) increases in the direction of \\(\nabla J\\). The chart below shows a conceptual learning curve (return \\(J\\) over updates).

{{< chart type="line" palette="math" title="J(θ) over policy gradient updates (conceptual)" labels="0, 20, 40, 60, 80" data="10, 35, 60, 82, 95" xLabel="Update" yLabel="J(θ)" >}}

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

At \\(x=2\\), \\(y=4\\) and \\(dy/dx=4\\). The chart below shows \\(y = x^2\\) at a few points (parabola).

{{< chart type="line" palette="math" title="y = x² (and slope 2x at x=2 is 4)" labels="0, 1, 2, 3, 4" data="0, 1, 4, 9, 16" xLabel="x" yLabel="y" >}}

---

6. **By hand:** For \\(f(x) = x^2 e^x\\), compute \\(f'(x)\\) using the product rule. Evaluate at \\(x = 0\\).

{{< collapse summary="Answer and explanation" >}}
**Product rule:** \\((uv)' = u'v + uv'\\). With \\(u = x^2\\) and \\(v = e^x\\), we have \\(u' = 2x\\) and \\(v' = e^x\\). So \\(f'(x) = 2x \cdot e^x + x^2 \cdot e^x = e^x(2x + x^2)\\).

**At \\(x = 0\\):** \\(f'(0) = e^0(0 + 0) = 0\\).

**Explanation:** The product rule is used whenever the loss or objective is a product of terms (e.g. policy probability times advantage). At \\(x=0\\) the slope is 0.

**Python:** `import numpy as np; x = 0; f = x**2 * np.exp(x); df = np.exp(x)*(2*x + x**2); print(df)` → 0. Or with PyTorch: `x = torch.tensor(0., requires_grad=True); (x**2 * torch.exp(x)).backward(); x.grad` → 0.
{{< /collapse >}}

\\(f'(x) = e^x(2x + x^2)\\); at \\(x=0\\) we get \\(f'(0)=0\\). The chart below shows \\(f'(x)\\) at a few points.

{{< chart type="line" palette="math" title="f'(x) = eˣ(2x + x²)" labels="-1, 0, 1, 2" data="-0.37, 0, 2.72, 8.15" xLabel="x" yLabel="f'(x)" >}}

---

7. **RL:** In **policy gradients**, we often compute \\(\nabla_\theta \log \pi(a|s;\theta)\\). Why is the *log* used instead of \\(\nabla_\theta \pi(a|s)\\)? (Hint: log-derivative trick; \\(\nabla \pi = \pi \nabla \log \pi\\), which keeps the gradient scaled by the probability.)

{{< collapse summary="Answer and explanation" >}}
**Log-derivative trick:** \\(\nabla_\theta \pi(a|s;\theta) = \pi(a|s;\theta) \cdot \nabla_\theta \log \pi(a|s;\theta)\\). So the gradient of the probability is the probability times the gradient of the log-probability.

**Why use log:** (1) The policy gradient theorem involves \\(\nabla_\theta \log \pi\\) weighted by the advantage; the log form keeps the gradient scaled by the probability and leads to a simple unbiased estimator. (2) In practice, \\(\log \pi\\) is numerically stable (we often work with log-probabilities to avoid underflow). (3) The resulting gradient update has a clean form (e.g. REINFORCE: increase log-probability of actions that got high return).

**Explanation:** So we use \\(\nabla_\theta \log \pi\\) in the update; the “trick” shows this is equivalent to a probability-weighted gradient of \\(\pi\\), which is what the policy gradient theorem requires.
**Python:** In PyTorch we compute `log_prob = policy.log_prob(action)` and then `loss = -log_prob * advantage`; `loss.backward()` gives the gradient of log π scaled by the advantage.
{{< /collapse >}}

The log-derivative trick: \\(\nabla \pi = \pi \nabla \log \pi\\). The chart below shows a conceptual comparison (gradient magnitude with vs without log).

{{< chart type="bar" palette="math" title="Gradient scale: π ∇log π vs ∇π (conceptual)" labels="∇log π (used), ∇π" data="1, 0.3" yLabel="Magnitude" >}}

---

## Professor's hints

- The chain rule is why we can train deep networks: we break the gradient into local pieces (each layer’s derivative) and multiply them. Autograd libraries do this automatically.
- In RL, “gradient” almost always means “with respect to the policy or value function parameters.” The reward is usually not differentiated; we differentiate the log-probability or the value prediction.
- When you see \\(\nabla_\theta \log \pi(a|s;\theta)\\), that is the gradient of the log-probability of the action under the policy—this vector appears in the policy gradient theorem.

---

---

*(Additional exercises — graded from drill to applied)*

8. **Drill — Power rule:** Differentiate f(x) = x³ + 2x - 5 with respect to x. Evaluate f'(2).

{{< collapse summary="Answer" >}}
f'(x) = 3x² + 2. At x=2: f'(2) = 3×4 + 2 = **14**.

**Python:** Numerical check: `h=1e-5; f=lambda x: x**3+2*x-5; print((f(2+h)-f(2-h))/(2*h))` → ≈14.
{{< /collapse >}}

---

9. **Apply — TD gradient:** The TD loss for a linear V is L(w) = ½(δ)² where δ = r + γ w·φ(s') - w·φ(s). Compute ∂L/∂w (ignoring the gradient through the target — semi-gradient).

{{< collapse summary="Answer" >}}
Semi-gradient: treat the target r + γ w·φ(s') as constant.

∂L/∂w = ∂/∂w [½(r + γ w·φ(s') - w·φ(s))²]  ← target treated as constant
= δ × (-φ(s))   (chain rule, derivative of w·φ(s) w.r.t. w is φ(s))

**Update:** w ← w - α × ∂L/∂w = w + α × δ × φ(s). This is the semi-gradient TD update.

**Why semi-gradient?** The true gradient would also differentiate through the target w.r.t. w. Semi-gradient ignores that term — it's a simplification that still converges for linear function approximation.
{{< /collapse >}}

---

10. **Drill — Chain rule:** f(x) = (2x + 1)³. Compute f'(x) using the chain rule. Evaluate at x=0.

{{< collapse summary="Answer" >}}
Let u = 2x + 1. f = u³. f'(x) = 3u² × u' = 3(2x+1)² × 2 = 6(2x+1)².

At x=0: f'(0) = 6(1)² = **6**.

**Python:** `h=1e-5; f=lambda x:(2*x+1)**3; print((f(h)-f(-h))/(2*h))` → ≈6.
{{< /collapse >}}

---

11. **Apply — Policy gradient sign:** The REINFORCE gradient is ∇J(θ) ∝ G_t × ∇log π(a_t|s_t;θ).

For a softmax policy over 2 actions, log π(a=0|s;θ) = θ_0 - log(exp(θ_0)+exp(θ_1)).

Compute ∂log π(a=0)/∂θ_0 and ∂log π(a=0)/∂θ_1 (in terms of π_0, π_1).

{{< collapse summary="Answer" >}}
log π(a=0) = θ_0 - log(exp(θ_0)+exp(θ_1)).

∂/∂θ_0: 1 - exp(θ_0)/(exp(θ_0)+exp(θ_1)) = 1 - π_0 = **π_1** (or equivalently 1 - π_0).

∂/∂θ_1: 0 - exp(θ_1)/(exp(θ_0)+exp(θ_1)) = **-π_1**.

So ∇log π(a=0) = [1-π_0, -π_1] = [π_1, -π_1].

**Interpretation:** A positive gradient step on action 0 increases θ_0 and decreases θ_1, increasing π_0 and decreasing π_1 — exactly what you want when action 0 gets a positive reward signal.
{{< /collapse >}}

---

12. **Drill — Partial derivatives:** f(x, y) = 3x²y + xy³. Compute ∂f/∂x and ∂f/∂y. Evaluate at (1, 2).

{{< collapse summary="Answer" >}}
∂f/∂x = 6xy + y³. At (1,2): 6×2 + 8 = **20**.

∂f/∂y = 3x² + 3xy². At (1,2): 3 + 6 = **9**.

**Why this matters:** Loss functions in deep RL depend on many parameters (weights). Backprop computes partial derivatives of the loss with respect to each weight, exactly this operation across thousands of parameters.
{{< /collapse >}}

---

13. **Apply — Clipped PPO gradient:** PPO clips the ratio r(θ) = π_θ(a)/π_old(a) at [1-ε, 1+ε]. The objective is J = min(r·A, clip(r,1-ε,1+ε)·A).

If A > 0 and r = 1.3 with ε = 0.2 (clip at 1.2): what is the clipped objective? What is ∂J/∂r at r=1.3?

{{< collapse summary="Answer" >}}
clip(1.3, 0.8, 1.2) = 1.2. J = min(1.3×A, 1.2×A) = **1.2×A** (clipped value is smaller).

∂J/∂r at r=1.3: since the clip is active (r > 1+ε), the minimum is the clipped term 1.2A, which has **zero gradient with respect to r**. The clip prevents further policy updates that would push r even higher.

**Insight:** PPO's clipping creates a "flat region" in the objective where the gradient is zero — exactly when the new policy deviates too much from the old one. This prevents large, destabilizing updates.
{{< /collapse >}}

---

14. **Think — Why minimize the squared TD error?** The TD update minimizes ½(δ)² where δ = r + γV(s') - V(s). Why square the TD error rather than minimize |δ|?

{{< collapse summary="Answer" >}}
Squaring the error: (1) makes it differentiable everywhere (|δ| has a non-differentiable kink at δ=0); (2) penalizes large errors more than small ones (squared grows faster); (3) the gradient is simply δ × (-1) = -δ, which is clean and easy to implement.

MSE loss is standard in regression; minimizing ½δ² leads to the TD update rule through one gradient step.

**Alternative:** Huber loss (L1 for large |δ|, L2 for small |δ|) is more robust to outliers and used in some DQN implementations.
{{< /collapse >}}

---

15. **Apply — Entropy gradient:** The entropy of a discrete distribution is H(π) = -Σ_a π(a) log π(a). Compute ∂H/∂π(a) for a fixed action a.

{{< collapse summary="Answer" >}}
∂H/∂π(a) = -log π(a) - π(a) × (1/π(a)) = **-(log π(a) + 1)**.

**In SAC:** The maximum entropy objective adds αH(π) to the reward. Its gradient -(log π(a) + 1) encourages the policy to keep action probabilities from collapsing to 0 (high entropy = more uniform = more exploration).

**Python (numerical check):**
```python
import math, numpy as np
pi = np.array([0.2, 0.5, 0.3])
# Gradient of entropy at action 0
print(-(math.log(pi[0]) + 1))   # ≈ 2.609
```
{{< /collapse >}}

{{< pyrepl code="import math\nimport numpy as np\n\npi = np.array([0.2, 0.5, 0.3])\nH = -sum(p * math.log(p) for p in pi)\nprint(f'Entropy H(pi) = {H:.4f}')\n\n# Gradient at each action\ngrads = [-(math.log(p) + 1) for p in pi]\nprint('dH/d_pi:', [round(g, 3) for g in grads])" height="220" >}}

---

## Common pitfalls

- **Maximize vs minimize:** Policy gradient *maximizes* return, so the update is \\(\theta \leftarrow \theta + \alpha \nabla_\theta J\\) (plus, not minus). Loss minimization uses minus. Mixing them up flips the direction of learning.
- **Forgetting the inner derivative in the chain rule:** For \\(f(g(x))\\), you must multiply by \\(g'(x)\\). For example, \\(\frac{d}{dx}e^{x^2} = e^{x^2} \cdot 2x\\).
- **Treating constants as variables:** When you differentiate with respect to \\(\theta\\), quantities that do not depend on \\(\theta\\) (e.g. rewards already observed) are constants; their derivative is zero. Only the part that depends on \\(\theta\\) (e.g. \\(\log \pi(a|s;\theta)\\)) contributes to the gradient.
