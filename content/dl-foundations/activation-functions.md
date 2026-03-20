---
title: "Activation Functions: Adding Non-Linearity"
description: "ReLU, sigmoid, tanh, and softmax — what they compute, when to use each, and why non-linearity is essential for deep networks."
date: 2026-03-20T00:00:00Z
weight: 3
draft: false
difficulty: 5
tags: ["activation functions", "ReLU", "sigmoid", "tanh", "softmax", "dl-foundations"]
keywords: ["ReLU", "sigmoid activation", "tanh", "softmax", "activation functions neural network", "dead ReLU"]
roadmap_icon: "sparkles"
roadmap_color: "green"
roadmap_phase_label: "Chapter 3"
---

**Learning objectives**

- Implement ReLU, sigmoid, tanh, and softmax in NumPy and state their formulas.
- Explain why activation functions are necessary — without them, all layers collapse to one linear transformation.
- Identify which activation to use for hidden layers versus output layers in RL networks.

**Concept and real-world motivation**

Without activation functions, a neural network is just a sequence of matrix multiplications — which collapses to a single matrix multiplication no matter how many layers you stack. Mathematically: \\(W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)\\), which is just \\(Ax + c\\) for some matrix \\(A\\) and vector \\(c\\). A ten-layer linear network is no more expressive than a one-layer linear network. **Activation functions break this linearity**, allowing networks to represent complex, non-linear functions.

The four most important activations:

- **ReLU** \\(f(z) = \max(0, z)\\): the workhorse of deep learning. Fast, sparse, avoids vanishing gradients. Default for hidden layers.
- **Sigmoid** \\(\sigma(z) = \frac{1}{1+e^{-z}}\\): squashes output to (0,1). Used for binary classification outputs. Prone to vanishing gradients in deep networks.
- **Tanh** \\(\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\\): squashes to (−1, 1). Zero-centered, which is better than sigmoid for hidden layers. Still suffers from vanishing gradients for very large |z|.
- **Softmax** \\(\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\\): converts a vector of scores into probabilities (sum to 1). Used for multi-class classification outputs and **policy networks in RL**.

**In RL networks: ReLU is the standard activation for hidden layers.** The output layer depends on the task: for DQN, the output is unbounded Q-values (no activation, or linear output). For policy gradient methods, the output is action probabilities — softmax over actions. The critic network outputs a scalar value estimate — also linear output.

**Illustration:**

{{< chart type="line" palette="math" title="ReLU values for z = −3 to 3" labels="-3, -2, -1, 0, 1, 2, 3" data="0, 0, 0, 0, 1, 2, 3" xLabel="z" yLabel="ReLU(z)" >}}

**Exercise:** Implement all four activation functions in NumPy. Compute and print ReLU, sigmoid, and tanh on an array of values, then compute softmax on a logit vector.

{{< pyrepl code="import numpy as np\n\n# Test values\nz = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])\n\n# TODO: implement ReLU\ndef relu(z):\n    pass  # hint: np.maximum(0, z)\n\n# TODO: implement sigmoid\ndef sigmoid(z):\n    pass  # hint: 1 / (1 + np.exp(-z))\n\n# TODO: implement tanh (np.tanh works, but also write the formula)\ndef tanh_manual(z):\n    pass  # hint: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))\n\n# TODO: implement softmax\ndef softmax(z):\n    pass  # hint: exp(z) / sum(exp(z)); use np.exp\n\nprint('z          :', z)\nprint('relu(z)    :', relu(z))\nprint('sigmoid(z) :', np.round(sigmoid(z), 4))\nprint('tanh(z)    :', np.round(tanh_manual(z), 4))\n\nlogits = np.array([1.0, 2.0, 3.0])\nprobs = softmax(logits)\nprint('\\nsoftmax([1,2,3]):', np.round(probs, 4))\nprint('sum of probs:', np.sum(probs))\n# expected: sum of probs = 1.0000" height="300" >}}

**Professor's hints**

- ReLU: `np.maximum(0, z)` (not `np.max`). `np.maximum` is element-wise; `np.max` returns a single value.
- Softmax: `exp_z = np.exp(z); return exp_z / np.sum(exp_z)`.
- Tanh is available as `np.tanh(z)` — use the manual formula to understand it, then verify with `np.tanh`.
- The sigmoid output at z=0 is exactly 0.5. The tanh output at z=0 is exactly 0.

**Common pitfalls**

- **`np.max` vs `np.maximum` in ReLU:** `np.maximum(0, z)` returns an array of the same shape as z. `np.max(z)` returns a single scalar.
- **Softmax numerical instability:** Computing `np.exp(z)` for large z (e.g., z=1000) overflows to `inf`. The fix is to subtract the maximum first: `exp_z = np.exp(z - np.max(z))`.
- **Softmax for a scalar:** Softmax is defined for vectors. Applying it to a single number always returns 1.0 — which is trivially correct but useless.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh_manual(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # numerically stable
    return exp_z / np.sum(exp_z)

print('relu(z)    :', relu(z))           # [0, 0, 0, 1, 3]
print('sigmoid(z) :', sigmoid(z).round(4))  # [0.0474, 0.2689, 0.5, 0.7311, 0.9526]
print('tanh(z)    :', tanh_manual(z).round(4))  # [-0.9951, -0.7616, 0.0, 0.7616, 0.9951]

logits = np.array([1.0, 2.0, 3.0])
probs = softmax(logits)
print('softmax:', probs.round(4))   # [0.0900, 0.2447, 0.6652]
print('sum:', np.sum(probs))         # 1.0
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Compute softmax by hand for `[1, 2, 3]`. Step 1: compute \\(e^1, e^2, e^3\\). Step 2: sum them. Step 3: divide each by the sum. Check with your implementation.

{{< pyrepl code="import numpy as np\n\n# Compute softmax([1, 2, 3]) step by step\nz = np.array([1.0, 2.0, 3.0])\n\n# TODO: step 1 — compute exp of each element\nexp_z = None\n\n# TODO: step 2 — sum all exponentials\ntotal = None\n\n# TODO: step 3 — divide each exp by the total\nprobs = None\n\nprint('exp_z:', exp_z)\nprint('total:', total)\nprint('probs:', probs)\nprint('sum of probs:', np.sum(probs))\n# expected: probs ≈ [0.0900, 0.2447, 0.6652], sum = 1.0" height="200" >}}

2. **Coding:** The "dead ReLU" problem: if z is always negative, ReLU always outputs 0 and the gradient is always 0 — the neuron never learns. Demonstrate this with `z = np.array([-5.0, -3.0, -2.0, -1.0])`. Compute ReLU and its gradient (1 where z > 0, else 0).
3. **Challenge:** Softmax is sensitive to the scale of its inputs. Compute `softmax([1, 2, 3])`, `softmax([10, 20, 30])`, and `softmax([0.1, 0.2, 0.3])`. Describe how scaling changes the "peakiness" of the output distribution. How does this relate to the temperature parameter in RL exploration?
4. **Variant:** Leaky ReLU: \\(f(z) = z\\) if \\(z > 0\\), else \\(0.01 z\\). This avoids dead neurons. Implement it and compare its output to ReLU for `z = [-3, -1, 0, 1, 3]`.

{{< pyrepl code="import numpy as np\n\nz = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])\n\n# TODO: implement leaky ReLU (alpha=0.01)\ndef leaky_relu(z, alpha=0.01):\n    pass  # hint: np.where(z > 0, z, alpha * z)\n\ndef relu(z):\n    return np.maximum(0, z)\n\nprint('z:          ', z)\nprint('relu:       ', relu(z))\nprint('leaky_relu: ', leaky_relu(z))\n# expected: leaky_relu = [-0.03, -0.01, 0.00, 1.00, 3.00]" height="200" >}}

5. **Debug:** The softmax below forgets to subtract the maximum before exponentiating, causing overflow for large inputs. Fix the numerical stability bug.

{{< pyrepl code="import numpy as np\n\n# BUG: no max subtraction — will overflow for large inputs\ndef softmax_buggy(z):\n    exp_z = np.exp(z)       # TODO: subtract np.max(z) first for stability\n    return exp_z / np.sum(exp_z)\n\n# Test with normal inputs (works)\nprint(softmax_buggy(np.array([1.0, 2.0, 3.0])).round(4))\n\n# Test with large inputs (overflows without the fix)\nlarge_z = np.array([1000.0, 1001.0, 1002.0])\nresult = softmax_buggy(large_z)\nprint(result)  # expected after fix: approximately [0.0900, 0.2447, 0.6652]" height="200" >}}

6. **Conceptual:** Why is tanh often preferred over sigmoid for hidden layers? Think about the output range (sigmoid: 0 to 1, tanh: −1 to 1) and what this means for the gradient flow during backpropagation.
7. **Recall:** Name all four activation functions covered in this page, write their formulas from memory, and state one use case for each (where in a neural network and for what task).
