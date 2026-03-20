---
title: "Biological Inspiration: From Brain Neurons to Artificial Neurons"
description: "How the biological neuron — dendrites, soma, axon — maps onto the artificial neuron with inputs, weights, bias, and activation."
date: 2026-03-20T00:00:00Z
weight: 1
draft: false
difficulty: 5
tags: ["biological neuron", "artificial neuron", "weights", "bias", "dl-foundations"]
keywords: ["biological neuron", "artificial neuron", "perceptron", "activation function", "neural network building block"]
roadmap_icon: "brain"
roadmap_color: "blue"
roadmap_phase_label: "Chapter 1"
---

**Learning objectives**

- Describe the structural parallel between a biological neuron and an artificial neuron.
- Compute the output of an artificial neuron given inputs, weights, bias, and an activation function.
- Implement a single neuron in NumPy using both the step function and the sigmoid activation.

**Concept and real-world motivation**

The human brain contains roughly 86 billion neurons. Each neuron receives electrical signals through branching extensions called **dendrites**, integrates those signals in the cell body (**soma**), and fires an output signal down its **axon** if the total input exceeds a threshold. This is an elegant natural computer: many inputs, weighted by synaptic strength, summed, then thresholded.

Warren McCulloch and Walter Pitts formalized this idea mathematically in 1943: the **artificial neuron** takes a vector of inputs \\(x = [x_1, x_2, \ldots, x_n]\\), multiplies each by a learned **weight** \\(w_i\\), adds a **bias** \\(b\\), and passes the result through an **activation function** \\(f\\). The output is:

\\[z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{w} \cdot \mathbf{x} + b\\]
\\[\text{output} = f(z)\\]

The weights encode how much each input matters. The bias shifts the threshold. The activation function determines whether the neuron "fires." In deep RL, this computation runs millions of times per second: **in DQN, the entire neural network IS the Q-function**: \\(Q(s, a; \theta) = \text{neural\_net}(s)[a]\\). Every weight \\(\theta\\) is updated during training via backpropagation to better predict future rewards.

**Illustration:**

{{< mermaid >}}
graph LR
    x1["x₁ = 0.5"] -->|"w₁ = 0.2"| N["Σ + activation"]
    x2["x₂ = 0.3"] -->|"w₂ = −0.5"| N
    x3["x₃ = 0.8"] -->|"w₃ = 0.4"| N
    bias["bias b = 0.1"] -->|"+"| N
    N --> out["output f(z)"]
{{< /mermaid >}}

**Exercise:** Implement a single artificial neuron in NumPy. Given the inputs, weights, and bias below, compute the pre-activation \\(z = \mathbf{w} \cdot \mathbf{x} + b\\), then apply (a) a step function and (b) the sigmoid function.

{{< pyrepl code="import numpy as np\n\n# Inputs, weights, bias\nx = np.array([0.5, 0.3, 0.8])\nw = np.array([0.2, -0.5, 0.4])\nb = 0.1\n\n# TODO: compute z = dot product of w and x, plus bias\nz = None  # replace with correct computation\n\n# TODO: step function activation: 1 if z > 0 else 0\nstep_output = None\n\n# TODO: sigmoid activation: 1 / (1 + exp(-z))\nsigmoid_output = None\n\nprint(f'z = {z:.4f}')\nprint(f'step output = {step_output}')\nprint(f'sigmoid output = {sigmoid_output:.4f}')\n# expected: z = 0.2700, step = 1, sigmoid ≈ 0.5671" height="240" >}}

**Professor's hints**

- The dot product \\(\mathbf{w} \cdot \mathbf{x}\\) is `np.dot(w, x)` — element-wise multiply then sum.
- The step function is simply `int(z > 0)` or `1 if z > 0 else 0`.
- `np.exp(-z)` computes \\(e^{-z}\\). The sigmoid is `1 / (1 + np.exp(-z))`.
- Check your answer: \\(z = 0.2(0.5) + (-0.5)(0.3) + 0.4(0.8) + 0.1 = 0.1 - 0.15 + 0.32 + 0.1 = 0.37\\). Wait — let me recount: 0.10 − 0.15 + 0.32 + 0.10 = 0.27. So z = 0.27.

**Common pitfalls**

- **Using a Python loop instead of `np.dot`:** `sum(w[i]*x[i] for i in range(3))` is correct but slow and not idiomatic NumPy. Use `np.dot(w, x)`.
- **Forgetting the bias:** \\(z = \mathbf{w} \cdot \mathbf{x}\\) without adding \\(b\\) is a common omission. The bias is the neuron's threshold offset.
- **Confusing sigmoid output with a probability:** Sigmoid output is in (0,1) and can be interpreted as a probability, but only when the network is trained with cross-entropy loss for a binary classification task.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

x = np.array([0.5, 0.3, 0.8])
w = np.array([0.2, -0.5, 0.4])
b = 0.1

z = np.dot(w, x) + b                    # 0.2*0.5 + (-0.5)*0.3 + 0.4*0.8 + 0.1
step_output = 1 if z > 0 else 0
sigmoid_output = 1 / (1 + np.exp(-z))

print(f'z = {z:.4f}')                   # z = 0.2700
print(f'step output = {step_output}')   # step output = 1
print(f'sigmoid output = {sigmoid_output:.4f}')  # sigmoid output ≈ 0.5671
```

The pre-activation \\(z = 0.2(0.5) + (-0.5)(0.3) + 0.4(0.8) + 0.1 = 0.10 - 0.15 + 0.32 + 0.10 = 0.27\\).

Since \\(z = 0.27 > 0\\), the step function outputs 1. The sigmoid maps 0.27 to \\(\frac{1}{1+e^{-0.27}} \approx 0.567\\), which is slightly above 0.5.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Change the weights and observe how the output changes. Try `w = [1.0, 1.0, 1.0]` — what is z? Try `w = [-1.0, -1.0, -1.0]` — what happens to the step output?

{{< pyrepl code="import numpy as np\n\nx = np.array([0.5, 0.3, 0.8])\nb = 0.1\n\n# Try different weight vectors\nfor w in [np.array([1.0, 1.0, 1.0]),\n          np.array([-1.0, -1.0, -1.0]),\n          np.array([0.0, 0.0, 0.0])]:\n    z = np.dot(w, x) + b\n    step = 1 if z > 0 else 0\n    sig  = 1 / (1 + np.exp(-z))\n    print(f'w={w}, z={z:.3f}, step={step}, sigmoid={sig:.3f}')" height="180" >}}

2. **Coding:** A neuron with 5 inputs. Initialize weights with `np.random.seed(7); w = np.random.randn(5)` and bias `b = np.random.randn()`. Generate input `x = np.array([1.0, 0.0, -0.5, 0.8, 0.2])`. Compute z and both activations.
3. **Challenge:** The step function has a gradient of 0 everywhere except at z=0. Why is this a problem for training with backpropagation? Explain why sigmoid is preferred over step for learning.
4. **Variant:** Implement the ReLU activation: \\(f(z) = \max(0, z)\\). Apply it to the same neuron above. For which values of z does ReLU give the same output as the step function?

{{< pyrepl code="import numpy as np\n\nx = np.array([0.5, 0.3, 0.8])\nw = np.array([0.2, -0.5, 0.4])\nb = 0.1\n\nz = np.dot(w, x) + b\n\n# TODO: implement ReLU\nrelu_output = None  # max(0, z)\n\nprint(f'z = {z:.4f}')\nprint(f'ReLU output = {relu_output:.4f}')\n# expected: ReLU output = 0.2700 (since z > 0)" height="180" >}}

5. **Debug:** The neuron below uses element-wise addition instead of a dot product to combine weights and inputs. Find and fix the bug.

{{< pyrepl code="import numpy as np\n\nx = np.array([0.5, 0.3, 0.8])\nw = np.array([0.2, -0.5, 0.4])\nb = 0.1\n\n# BUG: wrong operation for computing weighted sum\nz = w + x + b   # TODO: fix this line\n\nsigmoid = 1 / (1 + np.exp(-z))\nprint(f'z = {z}')           # expected: z = 0.2700 (a scalar)\nprint(f'sigmoid = {sigmoid}')  # expected: sigmoid ≈ 0.5671 (a scalar)" height="180" >}}

6. **Conceptual:** In the biological analogy, what does the bias \\(b\\) represent? How does increasing the bias affect how easily the neuron "fires"? What would a very large negative bias mean for the neuron?
7. **Recall:** Write the equation for an artificial neuron's output from memory, defining each symbol. Then write the sigmoid formula from memory. Check your answer against the page.
