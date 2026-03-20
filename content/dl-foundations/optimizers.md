---
title: "Optimizers: SGD, Momentum, and Adam"
description: "Understand SGD, Momentum, and Adam optimizers from scratch. Implement and compare them in NumPy."
date: 2026-03-20T00:00:00Z
weight: 8
draft: false
difficulty: 5
tags: ["deep learning", "optimizers", "SGD", "Adam", "momentum", "dl-foundations"]
keywords: ["SGD optimizer", "Adam optimizer", "momentum", "learning rate", "gradient descent", "optimizer comparison"]
roadmap_icon: "rocket"
roadmap_color: "violet"
roadmap_phase_label: "Chapter 8"
---

**Learning objectives**
- Implement SGD, Momentum, and Adam from scratch in NumPy
- Understand how learning rate affects convergence speed and stability
- Explain why adaptive optimizers like Adam often outperform plain SGD in practice
- Recognize that all RL gradient methods are variants of these optimizers

**Concept and real-world motivation**

An **optimizer** controls how weights are updated after each gradient computation. The simplest optimizer, **Stochastic Gradient Descent (SGD)**, moves weights in the direction opposite to the gradient by a fixed step size (learning rate \\(\alpha\\)). However, SGD can oscillate or converge slowly in valleys with steep walls and shallow floors.

**Momentum** adds a "velocity" term that accumulates past gradients, allowing the optimizer to build up speed in consistent directions and damp oscillations. Think of a ball rolling downhill — it picks up momentum instead of stopping at every bump. **Adam** goes further by keeping per-parameter adaptive learning rates: parameters that receive large gradients get smaller updates, and rarely updated parameters get larger updates. In practice, Adam is the default for most deep learning and RL work.

**In RL:** DQN uses Adam. Policy gradient methods use Adam or RMSprop. The TD loss or policy gradient is the "gradient" these optimizers receive. Choosing a learning rate too large causes instability; too small causes painfully slow convergence.

**Math:**

SGD: \\(w \leftarrow w - \alpha \nabla L\\)

Momentum: \\(v \leftarrow \beta v - \alpha \nabla L\\), \\(w \leftarrow w + v\\)

Adam:
- \\(m \leftarrow \beta_1 m + (1-\beta_1)\nabla L\\) (first moment — mean)
- \\(v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2\\) (second moment — uncentered variance)
- Bias-corrected: \\(\hat{m} = m/(1-\beta_1^t)\\), \\(\hat{v} = v/(1-\beta_2^t)\\)
- \\(w \leftarrow w - \alpha \frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}\\)

**Illustration — Adam loss curve (50 steps):**

{{< chart type="line" palette="learning" title="Adam loss over 50 training steps" labels="10,20,30,40,50" data="0.8,0.6,0.4,0.25,0.15" xLabel="Step" yLabel="Loss" >}}

**Exercise:** Implement SGD and Momentum from scratch and compare them minimizing \\(f(w) = (w-3)^2\\).

{{< pyrepl code="import numpy as np\n\n# f(w) = (w-3)^2, gradient = 2*(w-3)\ndef f(w): return (w - 3) ** 2\ndef grad_f(w): return 2 * (w - 3)\n\n# SGD\nw_sgd = 0.0\nlr = 0.1\nsgd_history = [w_sgd]\nfor _ in range(30):\n    g = grad_f(w_sgd)\n    w_sgd = w_sgd - lr * g\n    sgd_history.append(w_sgd)\n\n# Momentum\nw_mom = 0.0\nv = 0.0\nbeta = 0.9\nmom_history = [w_mom]\nfor _ in range(30):\n    g = grad_f(w_mom)\n    v = beta * v - lr * g\n    w_mom = w_mom + v\n    mom_history.append(w_mom)\n\nprint('SGD final w:', round(sgd_history[-1], 4), '  loss:', round(f(sgd_history[-1]), 6))\nprint('Mom final w:', round(mom_history[-1], 4), '  loss:', round(f(mom_history[-1]), 6))\nprint('SGD w trajectory (every 5):', [round(w, 3) for w in sgd_history[::5]])\nprint('Mom w trajectory (every 5):', [round(w, 3) for w in mom_history[::5]])" height="320" >}}

**Professor's hints**
- The learning rate is the single most important hyperparameter. Try 10x larger and 10x smaller to see the effect.
- Momentum \\(\beta=0.9\\) means "remember 90% of previous velocity." Higher \\(\beta\\) (e.g. 0.99) gives more momentum but can overshoot.
- Adam's bias correction is important in early steps — without it, the first update would be much too small.
- In RL, Adam with `lr=3e-4` is a common default that works across many problems.

**Common pitfalls**
- Applying \\(\beta\\) to the gradient instead of the velocity vector (see debug exercise below).
- Forgetting to reset momentum state when you restart training or change the model architecture.
- Using the same learning rate for all problems — always tune it.

{{< collapse summary="Worked solution" >}}
SGD converges but more slowly. Momentum overshoots slightly then converges faster because it accelerates in the consistent downhill direction.

For Adam from scratch:
```python
import numpy as np

def adam(grad_fn, w_init=0.0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=30):
    w = w_init
    m, v = 0.0, 0.0
    history = [w]
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w)
    return w, history
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** One SGD step by hand: \\(f(w) = w^2\\), \\(w=2\\), \\(lr=0.1\\). Compute \\(\nabla f\\), then \\(w_{new}\\).
{{< pyrepl code="# One SGD step: f(w) = w^2, w=2, lr=0.1\n# gradient of w^2 = 2w\nw = 2.0\nlr = 0.1\ngrad = 2 * w  # TODO: compute gradient\nw_new = w - lr * grad  # TODO: apply update\nprint('gradient:', grad)\nprint('w_new:', w_new)  # expected: 1.6" height="180" >}}

2. **Coding:** Implement Adam from scratch (all 5 update equations). Test on \\(f(w)=(w-3)^2\\) and verify it converges faster than SGD in fewer steps.

3. **Challenge:** Extend to a 2D function \\(f(w_1, w_2) = w_1^2 + 10 w_2^2\\) (an elongated bowl). Compare SGD and Adam — why does Adam handle the different scales better?

4. **Variant:** Implement RMSprop: \\(v \leftarrow \rho v + (1-\rho)(\nabla L)^2\\), \\(w \leftarrow w - \frac{\alpha}{\sqrt{v}+\epsilon}\nabla L\\). RMSprop is Adam without momentum — compare them.

5. **Debug:** Fix the momentum bug below where \\(\beta\\) is applied to the gradient instead of the velocity:
{{< pyrepl code="# BUG: beta applied to gradient, not velocity\nw = 0.0\nv = 0.0\nlr = 0.1\nbeta = 0.9\nfor _ in range(20):\n    g = 2 * (w - 3)\n    v = beta * g - lr * g  # BUG: should be beta*v - lr*g\n    w = w + v\nprint('w (buggy):', round(w, 4), '  expected ~3.0')\n# TODO: fix the bug" height="200" >}}

6. **Conceptual:** Why does Adam use bias correction (\\(\hat{m}\\) and \\(\hat{v}\\))? What value would \\(m\\) take at step \\(t=1\\) if bias correction were omitted and \\(\beta_1=0.9\\), gradient=1?

7. **Recall:** In one sentence each: (a) What is the role of \\(\epsilon\\) in Adam? (b) Why do we need momentum? (c) What does a learning rate scheduler do?
