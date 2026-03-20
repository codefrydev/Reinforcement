---
title: "Checkpoint: DL Foundations Mid-Point"
description: "5 questions after completing the first 6 DL Foundations pages. Check your understanding before continuing."
date: 2026-03-20T00:00:00Z
draft: false
tags: ["assessment", "checkpoint", "deep learning", "dl-foundations"]
keywords: ["DL checkpoint", "neural network quiz", "forward pass check"]
weight: 6
roadmap_icon: "network"
roadmap_color: "purple"
roadmap_phase_label: "DL Mid-Point"
---

Take this checkpoint after completing the first 6 DL Foundations pages (Biological Inspiration through Loss Functions). All 5 should feel manageable — if any are unclear, re-read the relevant page before continuing.

---

**Q1.** Write the formula for a single artificial neuron output. Use \\(z\\) for the pre-activation and \\(a\\) for the output.

{{< collapse summary="Answer" >}}
**Pre-activation:** \\(z = w \cdot x + b = \sum_i w_i x_i + b\\) (dot product of weights and inputs, plus bias)

**Output:** \\(a = f(z)\\) where \\(f\\) is an activation function (e.g. ReLU, sigmoid, tanh).

For a layer with multiple neurons: \\(z = Wx + b\\) (matrix form), \\(a = f(z)\\) applied element-wise.
{{< /collapse >}}

---

**Q2.** What is XOR's significance in neural network history?

{{< collapse summary="Answer" >}}
XOR is the canonical example that **a single-layer perceptron cannot solve non-linearly separable problems**. Minsky & Papert's 1969 book showed this limitation, contributing to the first "AI winter" as funding dried up for neural network research.

The resolution: adding hidden layers creates non-linear decision boundaries. A 2-layer network (1 hidden layer) can solve XOR. This was a key insight behind the revival of neural networks in the 1980s with backpropagation.
{{< /collapse >}}

---

**Q3.** For \\(z=2\\): compute ReLU(z) and sigmoid(z).

{{< pyrepl code="import numpy as np\n\nz = 2.0\nrelu_z = max(0, z)\nsigmoid_z = 1 / (1 + np.exp(-z))\nprint(f'z = {z}')\nprint(f'ReLU(z) = {relu_z}')\nprint(f'sigmoid(z) = {sigmoid_z:.4f}')" height="160" >}}

{{< collapse summary="Answer" >}}
- **ReLU(2) = 2** — ReLU is the identity for positive inputs; \\(\max(0, 2) = 2\\).
- **sigmoid(2) ≈ 0.8808** — \\(\frac{1}{1+e^{-2}} \approx 0.8808\\).

Key difference: ReLU passes the value through unchanged (or zeroes it); sigmoid squashes it to (0,1). ReLU is preferred for hidden layers because it doesn't squash and avoids vanishing gradients.
{{< /collapse >}}

---

**Q4.** Forward pass: \\(x=[1,0]\\), \\(W=\begin{bmatrix}1&2\\3&4\end{bmatrix}\\), \\(b=[0,0]\\). Compute \\(h = \text{ReLU}(Wx + b)\\).

{{< collapse summary="Answer" >}}
\\(Wx = \begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}1 \cdot 1 + 2 \cdot 0 \\ 3 \cdot 1 + 4 \cdot 0\end{bmatrix} = \begin{bmatrix}1\\3\end{bmatrix}\\)

Adding bias \\(b=[0,0]\\): \\(z = [1, 3]\\).

\\(h = \text{ReLU}([1, 3]) = [1, 3]\\) — both positive, so ReLU leaves them unchanged.
{{< /collapse >}}

---

**Q5.** What is the advantage of cross-entropy loss over MSE for classification problems?

{{< collapse summary="Answer" >}}
**Cross-entropy** \\(L = -\sum_i y_i \log(\hat{y}_i)\\) has two key advantages over MSE for classification:

1. **Better gradient signal:** When a prediction is very wrong (e.g. predicting probability 0.01 for the true class), cross-entropy gives a large gradient (`-log(0.01) = 4.6`), pushing the model to correct quickly. MSE would give a smaller gradient in this case.

2. **Correct interpretation:** Cross-entropy directly measures the quality of predicted probability distributions (how many bits are needed to encode the true label given the prediction). MSE treats classification outputs as continuous values, which is conceptually wrong.

For regression, use MSE. For classification, use cross-entropy.
{{< /collapse >}}

---

All 5 correct? Continue to the remaining DL Foundations pages (Optimizers through Review & Bridge). Stuck on 2 or more? Re-read the pages covering the topics you missed.
