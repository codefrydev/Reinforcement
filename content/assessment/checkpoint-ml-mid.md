---
title: "Checkpoint: ML Foundations Mid-Point"
description: "5 questions after completing the first 7 ML Foundations pages. Check your understanding before continuing."
date: 2026-03-20T00:00:00Z
draft: false
tags: ["assessment", "checkpoint", "machine learning", "ml-foundations"]
keywords: ["ML checkpoint", "machine learning quiz", "supervised learning check"]
weight: 4
roadmap_icon: "chart"
roadmap_color: "amber"
roadmap_phase_label: "ML Mid-Point"
---

Take this checkpoint after completing the first 7 ML Foundations pages (What is ML through Logistic Regression). All 5 should feel manageable — if any are unclear, re-read the relevant page before continuing.

---

**Q1.** What is the difference between supervised and unsupervised learning? Give an example of each.

{{< collapse summary="Answer" >}}
**Supervised learning:** The training data includes input-output pairs (labels). The model learns to map inputs to outputs. Example: classifying emails as spam/not-spam (each email has a known label).

**Unsupervised learning:** The training data has no labels — only inputs. The model finds structure on its own. Example: clustering customers by purchasing behavior (no predefined groups).

**Reinforcement learning** (for comparison): an agent learns by interacting with an environment and receiving rewards — no labels, but feedback after actions.
{{< /collapse >}}

---

**Q2.** Write the MSE formula. For predictions [2, 4, 6] and true values [2.5, 3.5, 6.5], compute MSE.

{{< collapse summary="Answer" >}}
\\(\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2\\)

Errors: \\(2.5-2=0.5\\), \\(3.5-4=-0.5\\), \\(6.5-6=0.5\\). Squared: \\(0.25, 0.25, 0.25\\). MSE = \\(\frac{0.25+0.25+0.25}{3} = 0.25\\).
{{< /collapse >}}

---

**Q3.** What is gradient descent? Write the weight update rule.

{{< collapse summary="Answer" >}}
**Gradient descent** is an optimization algorithm that iteratively moves weights in the direction that reduces the loss. At each step, compute the gradient of the loss with respect to the weights, then subtract a small fraction (the learning rate) of the gradient from the current weights.

Weight update rule: \\(w \leftarrow w - \alpha \nabla_w L(w)\\)

Where \\(\alpha\\) is the learning rate (e.g. 0.01) and \\(\nabla_w L\\) is the gradient of the loss with respect to weights.
{{< /collapse >}}

---

**Q4.** What is the sigmoid function? What value does it output for z=0?

{{< collapse summary="Answer" >}}
\\(\sigma(z) = \frac{1}{1+e^{-z}}\\)

Sigmoid squashes any real number to \\((0, 1)\\), making it useful for binary classification (interpret output as probability of class 1).

For \\(z=0\\): \\(\sigma(0) = \frac{1}{1+e^0} = \frac{1}{1+1} = 0.5\\). The sigmoid is exactly 0.5 at z=0 — the decision boundary.
{{< /collapse >}}

---

**Q5.** Implement sigmoid and test it on the values below.

{{< pyrepl code="import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\nz = np.array([0, 1, -1, 2])\nresult = sigmoid(z)\nprint('sigmoid([0, 1, -1, 2]):', result.round(4))\n# expected: [0.5, 0.7311, 0.2689, 0.8808]" height="160" >}}

{{< collapse summary="Answer" >}}
`sigmoid(0) = 0.5`, `sigmoid(1) ≈ 0.731`, `sigmoid(-1) ≈ 0.269`, `sigmoid(2) ≈ 0.881`.

Key properties: sigmoid is symmetric around 0.5, approaches 1 as z→+∞, approaches 0 as z→−∞. Note that sigmoid(-z) = 1 - sigmoid(z).
{{< /collapse >}}

---

All 5 correct? Continue to the remaining ML Foundations pages. Stuck on 2 or more? Re-read the pages covering the topics you missed.
