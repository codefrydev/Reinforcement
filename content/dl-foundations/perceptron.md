---
title: "The Perceptron: Learning from Mistakes"
description: "The perceptron learning rule, training on AND and OR gates, and why XOR exposes the fundamental limitation of single-layer networks."
date: 2026-03-20T00:00:00Z
weight: 2
draft: false
difficulty: 5
tags: ["perceptron", "AND gate", "XOR", "linear separability", "dl-foundations"]
keywords: ["perceptron learning rule", "AND gate", "XOR problem", "linear separability", "single-layer network"]
roadmap_icon: "network"
roadmap_color: "teal"
roadmap_phase_label: "Chapter 2"
---

**Learning objectives**

- State the perceptron learning rule and implement it in NumPy.
- Train a perceptron on the AND gate and verify it achieves 100% accuracy.
- Explain why XOR is not linearly separable and why this requires multi-layer networks.

**Concept and real-world motivation**

Frank Rosenblatt's **perceptron** (1958) is the simplest possible neural network: one neuron, one layer, one learning rule. The perceptron takes binary inputs, computes a weighted sum plus bias, and outputs 1 if the result is positive, 0 otherwise. It learns by correcting its mistakes: whenever it predicts wrongly, it nudges its weights in the direction that would have produced the correct answer.

The **perceptron learning rule** updates weights only on misclassified examples:

\\[w \leftarrow w + \alpha (y - \hat{y}) x\\]
\\[b \leftarrow b + \alpha (y - \hat{y})\\]

where \\(\alpha\\) is the learning rate, \\(y\\) is the true label, and \\(\hat{y}\\) is the predicted label. When the prediction is correct, \\(y - \hat{y} = 0\\) and nothing changes. When wrong, the weight moves toward the correct classification.

The perceptron can learn any **linearly separable** problem — one where a straight line (or hyperplane in higher dimensions) separates the two classes. AND and OR are linearly separable. **XOR is not.** No single line can separate the XOR truth table into its two classes, which is why single-layer perceptrons cannot solve XOR. The fix is multiple layers — exactly what we build in the next page. In RL, this is why Q-networks need hidden layers: the mapping from state to Q-value is almost never linearly separable.

**Illustration:**

{{< chart type="bar" palette="math" title="AND gate: perceptron weights after training" labels="w1_final, w2_final, bias_final" data="0.4, 0.4, -0.6" xLabel="Parameter" yLabel="Value" >}}

**Exercise:** Implement the perceptron learning rule on the AND gate. Initialize all weights to zero, train for 10 epochs over the 4 training examples, and print the final weights and accuracy.

{{< pyrepl code="import numpy as np\n\n# AND gate training data: (input, label)\nX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\ny = np.array([0, 0, 0, 1])\n\n# TODO: initialize weights and bias to zero\nw = None  # shape (2,)\nb = None  # scalar\nalpha = 1.0\n\ndef predict(x, w, b):\n    # TODO: return 1 if w·x + b > 0 else 0\n    pass\n\n# TODO: train for 10 epochs\n# For each epoch, iterate over all (X[i], y[i])\n# If predict(X[i]) != y[i], update w and b\nfor epoch in range(10):\n    for i in range(len(X)):\n        y_hat = predict(X[i], w, b)\n        # TODO: apply perceptron update rule\n        pass\n\n# Evaluate\npreds = np.array([predict(X[i], w, b) for i in range(len(X))])\nacc = np.mean(preds == y)\nprint(f'Final weights: w={w}, b={b}')\nprint(f'Predictions: {preds}')\nprint(f'Accuracy: {acc:.0%}')\n# expected: Accuracy: 100%" height="300" >}}

**Professor's hints**

- Initialize `w = np.zeros(2)` and `b = 0.0`.
- The `predict` function: `z = np.dot(w, x) + b; return 1 if z > 0 else 0`.
- The update: `w += alpha * (y[i] - y_hat) * X[i]` and `b += alpha * (y[i] - y_hat)`.
- The perceptron convergence theorem guarantees convergence in finite steps for linearly separable data. AND is linearly separable, so 10 epochs is more than enough.
- After training, typical values are `w ≈ [0.4, 0.4]`, `b ≈ -0.6` (exact values depend on update order).

**Common pitfalls**

- **Updating on every example, not just mistakes:** The perceptron only updates when it's wrong. The condition is `if y_hat != y[i]`, not unconditional.
- **Using wrong learning rate:** For binary perceptrons, `alpha = 1.0` is standard. Fractional learning rates just slow convergence without changing the final result for linearly separable data.
- **Expecting exact weight values:** Many weight configurations correctly classify AND — the perceptron finds one valid solution, not a unique one.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w = np.zeros(2)
b = 0.0
alpha = 1.0

def predict(x, w, b):
    return 1 if np.dot(w, x) + b > 0 else 0

for epoch in range(10):
    for i in range(len(X)):
        y_hat = predict(X[i], w, b)
        if y_hat != y[i]:
            w += alpha * (y[i] - y_hat) * X[i]
            b += alpha * (y[i] - y_hat)

preds = np.array([predict(X[i], w, b) for i in range(len(X))])
print(f'Final weights: w={w}, b={b}')   # w=[1. 1.], b=-1.0 (or similar)
print(f'Accuracy: {np.mean(preds == y):.0%}')  # 100%
```

The perceptron finds a separating hyperplane: the line \\(x_1 + x_2 = 1\\) (or equivalently \\(w_1 x_1 + w_2 x_2 - 1 > 0\\)). Only the point (1,1) lies on the positive side, so AND is learned correctly.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Run the perceptron on OR gate data: \\(y = [0, 1, 1, 1]\\). Does it converge? Print the final accuracy.

{{< pyrepl code="import numpy as np\n\nX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\ny_or = np.array([0, 1, 1, 1])  # OR gate\n\nw = np.zeros(2)\nb = 0.0\nalpha = 1.0\n\ndef predict(x, w, b):\n    return 1 if np.dot(w, x) + b > 0 else 0\n\nfor epoch in range(10):\n    for i in range(len(X)):\n        y_hat = predict(X[i], w, b)\n        if y_hat != y_or[i]:\n            w += alpha * (y_or[i] - y_hat) * X[i]\n            b += alpha * (y_or[i] - y_hat)\n\npreds = np.array([predict(X[i], w, b) for i in range(len(X))])\nprint(f'OR gate accuracy: {np.mean(preds == y_or):.0%}')\n# expected: 100%" height="220" >}}

2. **Coding:** Try the XOR gate: `y_xor = [0, 1, 1, 0]`. Run the same perceptron for 100 epochs. Print accuracy every 20 epochs. Show that it never exceeds 75%.

{{< pyrepl code="import numpy as np\n\nX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\ny_xor = np.array([0, 1, 1, 0])  # XOR gate\n\nw = np.zeros(2)\nb = 0.0\nalpha = 1.0\n\ndef predict(x, w, b):\n    return 1 if np.dot(w, x) + b > 0 else 0\n\nfor epoch in range(100):\n    for i in range(len(X)):\n        y_hat = predict(X[i], w, b)\n        if y_hat != y_xor[i]:\n            w += alpha * (y_xor[i] - y_hat) * X[i]\n            b += alpha * (y_xor[i] - y_hat)\n    if (epoch + 1) % 20 == 0:\n        preds = np.array([predict(X[i], w, b) for i in range(len(X))])\n        acc = np.mean(preds == y_xor)\n        print(f'Epoch {epoch+1:3d}: accuracy={acc:.0%}')\n# expected: accuracy stays at 50% or 75% — never reaches 100%" height="240" >}}

3. **Challenge:** Draw the 2D decision boundary for a perceptron trained on AND. The boundary is the line \\(w_1 x_1 + w_2 x_2 + b = 0\\). For the weights you found, write this as \\(x_2 = f(x_1)\\). Does this line correctly separate the AND truth table?
4. **Variant:** Implement the perceptron on a 3-input AND gate: inputs are all 8 combinations of 3 bits, and output is 1 only when all three are 1. Does it still converge in 10 epochs?
5. **Debug:** The perceptron below has a wrong sign in the update rule — it moves weights in the wrong direction. Find and fix it.

{{< pyrepl code="import numpy as np\n\nX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\ny = np.array([0, 0, 0, 1])\n\nw = np.zeros(2)\nb = 0.0\nalpha = 1.0\n\ndef predict(x, w, b):\n    return 1 if np.dot(w, x) + b > 0 else 0\n\nfor epoch in range(20):\n    for i in range(len(X)):\n        y_hat = predict(X[i], w, b)\n        if y_hat != y[i]:\n            # BUG: wrong sign in update\n            w -= alpha * (y[i] - y_hat) * X[i]\n            b -= alpha * (y[i] - y_hat)\n\npreds = np.array([predict(X[i], w, b) for i in range(len(X))])\nprint(f'Accuracy: {np.mean(preds == y):.0%}')\n# expected after fix: 100%" height="220" >}}

6. **Conceptual:** The perceptron convergence theorem states that if the data is linearly separable, the perceptron will converge in a finite number of steps. What does "linearly separable" mean geometrically for 2D inputs? Draw a sketch and mark which truth tables (AND, OR, NAND, XOR) are linearly separable.
7. **Recall:** Write the perceptron update rule from memory. Then state the condition under which the update fires (when does the perceptron update its weights?). What happens when the perceptron predicts correctly?
