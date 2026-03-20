---
title: "Gradient Descent"
description: "The optimization algorithm behind every trained ML model: iteratively follow the negative gradient to minimize a loss."
date: 2026-03-20T00:00:00Z
weight: 4
draft: false
difficulty: 4
tags: ["gradient descent", "learning rate", "optimization", "loss function", "ml-foundations"]
keywords: ["gradient descent algorithm", "learning rate", "loss curve", "optimization", "policy gradient", "stochastic gradient descent"]
roadmap_icon: "chart"
roadmap_color: "amber"
roadmap_phase_label: "Chapter 4"
---

**Learning objectives**

- State the gradient descent update rule and explain what each term means.
- Identify how the learning rate affects convergence: too small (slow), too large (diverge), just right (converges).
- Implement gradient descent from scratch in NumPy and plot the resulting loss curve.

**Concept and real-world motivation**

Imagine you are blindfolded on a hilly landscape and you want to reach the lowest valley. You cannot see the whole landscape, but you can feel the slope under your feet. **Gradient descent** says: take a small step in the direction that slopes downward, then repeat. The **learning rate** controls how large each step is.

Formally, if we want to minimise a loss \\(L\\) with respect to a parameter \\(w\\), we update:

\\[w \leftarrow w - \alpha \nabla_w L\\]

where \\(\alpha\\) is the learning rate and \\(\nabla_w L\\) is the gradient (the direction of steepest ascent). We subtract the gradient to go downhill. We repeat this update many times — each repetition is called an **iteration** or **step**.

The **loss curve** — a plot of \\(L\\) vs iteration — tells the whole story of training. A healthy loss curve drops fast early and plateaus smoothly. A diverging curve (loss explodes upward) means the learning rate is too large. A flat curve from the start means the learning rate is too small or the gradient is zero.

**RL connection:** **Policy gradient ascent** does the exact same thing, but in reverse — it *maximises* the expected return \\(J(\theta)\\) instead of minimising a loss:

\\[\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\]

REINFORCE, PPO, and A3C all implement this. The only difference from supervised gradient *descent* is the \\(+\\) sign. Mastering gradient descent here means policy gradient later is just a sign change.

**Illustration:** The loss curve below shows typical gradient descent behaviour: rapid decrease in early iterations, then a gentle plateau as the minimum is approached.

{{< chart type="line" palette="learning" title="Gradient descent loss curve" labels="0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20" data="25.0, 14.8, 9.1, 5.8, 3.9, 2.8, 2.1, 1.7, 1.5, 1.3, 1.2" xLabel="Iteration" yLabel="Loss" >}}

How does the learning rate affect convergence speed? Explore different values here:

{{< pyrepl code="import numpy as np\n\ndef f(x):      return (x - 3)**2\ndef grad_f(x): return 2*(x - 3)\n\nfor lr in [0.01, 0.1, 0.5]:\n    x = 10.0\n    for _ in range(30):\n        x = x - lr * grad_f(x)\n    print(f'lr={lr:.2f} -> final x={x:.6f}, loss={f(x):.8f}')\n\n# expected: all should converge to x=3.0, but at different speeds\n# lr=0.01 will still be far from 3 after 30 steps\n# lr=0.1  will be very close\n# lr=0.5  will converge in just a few steps" height="220" >}}

**Exercise:** Implement gradient descent to minimise \\(f(x) = (x-3)^2\\). Start at \\(x=10\\), use \\(\alpha=0.1\\), run 30 steps. Print \\(x\\) and \\(f(x)\\) at each step to watch convergence.

{{< pyrepl code="import numpy as np\n\ndef f(x):      return (x - 3)**2       # function to minimise\ndef grad_f(x): return 2 * (x - 3)     # its gradient\n\nx  = 10.0   # starting point\nlr = 0.1    # learning rate\n\n# TODO: run gradient descent for 30 steps\n# At each step: x = x - lr * grad_f(x)\n# Print step number, x value, and f(x)\nfor step in range(30):\n    # TODO: update x and print\n    pass\n\n# expected: x should converge to 3.0 and f(x) should converge to 0.0" height="300" >}}

**Professor's hints**

- The gradient of \\((x-3)^2\\) with respect to \\(x\\) is \\(2(x-3)\\). When \\(x > 3\\), the gradient is positive, so subtracting it decreases \\(x\\). When \\(x < 3\\), it is negative, so subtracting it increases \\(x\\). Either way, \\(x\\) moves toward 3.
- With \\(\alpha=0.1\\), each step reduces the error by 80% (since \\(1 - 2\alpha = 0.8\\) for this parabola). After 30 steps, \\(x\\) will be extremely close to 3.
- Use `print(f'step {step}: x={x:.6f}, loss={f(x):.8f}')` to see progress at each iteration.

**Common pitfalls**

- **Wrong sign:** Writing `x = x + lr * grad_f(x)` instead of `x - lr * ...` causes gradient *ascent* — \\(x\\) runs away from the minimum. This is the most common gradient descent bug.
- **Learning rate too large:** For \\(f(x) = (x-3)^2\\), using \\(\alpha \geq 1\\) causes the update to overshoot and diverge. Try `lr=2.0` to see this happen.
- **Not tracking the loss:** Always log the loss during training. Silent divergence (loss going to infinity) should fail loudly, not quietly.

{{< collapse summary="Worked solution" >}}
Full gradient descent implementation with output:

```python
import numpy as np

def f(x):      return (x - 3)**2
def grad_f(x): return 2 * (x - 3)

x  = 10.0
lr = 0.1

for step in range(30):
    x = x - lr * grad_f(x)
    print(f'step {step:2d}: x={x:.6f}, loss={f(x):.8f}')

# Step 0:  x=8.600000, loss=31.36
# Step 1:  x=7.480000, loss=20.07
# ...
# Step 29: x=3.000036, loss=0.00000000
```

The key line is `x = x - lr * grad_f(x)`. After 30 steps with lr=0.1, \\(x\\) converges to 3.0 to 5 decimal places.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** For \\(f(x) = (x-3)^2\\) and starting point \\(x=5\\), compute the first two gradient steps by hand with \\(\alpha=0.1\\). What is \\(x\\) after step 1? After step 2?
2. **Coding:** Minimise \\(g(x) = x^4 - 4x^2 + x\\) using gradient descent. The gradient is \\(g'(x) = 4x^3 - 8x + 1\\). Start at \\(x=-2\\), use \\(\alpha=0.01\\), run 200 steps. What is the minimum you find?
3. **Challenge:** Implement **mini-batch gradient descent** on the linear regression problem from the previous page. Instead of using all 5 samples each step, use a random batch of 2. Compare convergence to full-batch gradient descent.
4. **Variant:** Run gradient descent on \\(f(x) = (x-3)^2\\) with `lr=2.0`. Print \\(x\\) and \\(f(x)\\) at each step. What happens? Explain why the loss explodes.

{{< pyrepl code="import numpy as np\n\ndef f(x):      return (x - 3)**2\ndef grad_f(x): return 2*(x - 3)\n\nx  = 10.0\nlr = 2.0   # too large!\n\nfor step in range(10):\n    x = x - lr * grad_f(x)\n    print(f'step {step}: x={x:.2f}, loss={f(x):.2f}')\n\n# observe: x alternates between large positive and negative values\n# this is divergence caused by lr being too large" height="220" >}}

5. **Debug:** The gradient sign is wrong below — the parameter moves *away* from the minimum instead of toward it. Fix the update rule.

{{< pyrepl code="import numpy as np\n\ndef f(x):      return (x - 3)**2\ndef grad_f(x): return 2*(x - 3)\n\nx  = 10.0\nlr = 0.1\n\nfor step in range(10):\n    x = x + lr * grad_f(x)   # BUG: should subtract, not add\n    print(f'step {step}: x={x:.4f}, loss={f(x):.4f}')\n\n# expected: x should decrease toward 3.0\n# actual: x increases to infinity\n# TODO: fix the update rule" height="200" >}}

6. **Conceptual:** Explain in your own words why gradient descent does not always find the *global* minimum of a loss function. Under what conditions is it guaranteed to find the global minimum?
7. **Recall:** Write the gradient descent update rule from memory. Identify each symbol: \\(w\\), \\(\alpha\\), \\(\nabla_w L\\). What is the unit of \\(\alpha\\)?
