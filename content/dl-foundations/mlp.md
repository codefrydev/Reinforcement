---
title: "Multi-Layer Perceptrons: Stacking Layers to Break Linearity"
description: "MLP architecture, parameter counting, and how stacking non-linear layers allows networks to solve XOR and approximate any function."
date: 2026-03-20T00:00:00Z
weight: 4
draft: false
difficulty: 5
tags: ["MLP", "multi-layer perceptron", "architecture", "parameters", "dl-foundations"]
keywords: ["multi-layer perceptron", "neural network architecture", "parameter counting", "XOR neural network", "hidden layers"]
roadmap_icon: "layers"
roadmap_color: "amber"
roadmap_phase_label: "Chapter 4"
---

**Learning objectives**

- Describe the architecture of a multi-layer perceptron and name each component.
- Count the total number of trainable parameters in an MLP given its layer sizes.
- Implement the forward pass of a small MLP in NumPy using pre-given weights and verify it solves XOR.

**Concept and real-world motivation**

A single perceptron can only draw a straight line through the input space — it can solve AND and OR, but not XOR. The solution is to stack multiple layers: each layer transforms the input into a new representation, and successive non-linear transformations can carve out arbitrarily complex decision boundaries.

A **multi-layer perceptron (MLP)** consists of:
- An **input layer** (no computation, just the raw features \\(x\\))
- One or more **hidden layers**, each applying \\(h = f(Wx + b)\\) where \\(W\\) is a weight matrix, \\(b\\) is a bias vector, and \\(f\\) is an activation function
- An **output layer** that produces the network's prediction

Each layer \\(\ell\\) has a weight matrix \\(W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}\\) and a bias vector \\(b_\ell \in \mathbb{R}^{n_\ell}\\). The **total parameter count** for one layer connecting \\(n_{in}\\) neurons to \\(n_{out}\\) neurons is \\(n_{out} \times n_{in} + n_{out}\\) (weights plus biases).

**The Q-network in DQN is typically a 3-layer MLP:** input layer = state representation, two hidden layers with 64 or 512 ReLU neurons, output layer = one Q-value per action. When Atari DQN processes pixel frames, a convolutional front-end precedes the MLP, but the final layers are exactly this structure.

**Architecture:**

{{< mermaid >}}
graph LR
    in["Input\n[x₁, x₂, x₃, x₄]"] --> h1["Hidden 1\n8 neurons (ReLU)"]
    h1 --> h2["Hidden 2\n4 neurons (ReLU)"]
    h2 --> out["Output\n2 neurons (softmax)"]
{{< /mermaid >}}

**Exercise:** Count the parameters in a 3-layer MLP, then initialize the weight matrices and bias vectors with `np.random.randn`. Verify the shapes are correct.

{{< pyrepl code="import numpy as np\nnp.random.seed(42)\n\n# Architecture: 4 inputs → 8 hidden → 4 hidden → 2 outputs\nn_input  = 4\nn_hidden1 = 8\nn_hidden2 = 4\nn_output = 2\n\n# TODO: count parameters in each layer\n# Layer 1: W1 shape = (n_hidden1, n_input), b1 shape = (n_hidden1,)\n# Layer 2: W2 shape = (n_hidden2, n_hidden1), b2 shape = (n_hidden2,)\n# Layer 3: W3 shape = (n_output, n_hidden2), b3 shape = (n_output,)\n\nparams_layer1 = None  # TODO: n_hidden1 * n_input + n_hidden1\nparams_layer2 = None  # TODO: n_hidden2 * n_hidden1 + n_hidden2\nparams_layer3 = None  # TODO: n_output * n_hidden2 + n_output\ntotal_params  = None  # TODO: sum of all three\n\nprint(f'Layer 1 params: {params_layer1}')\nprint(f'Layer 2 params: {params_layer2}')\nprint(f'Layer 3 params: {params_layer3}')\nprint(f'Total params:   {total_params}')\n\n# TODO: initialize weight matrices and bias vectors\nW1 = np.random.randn(n_hidden1, n_input)\nb1 = np.random.randn(n_hidden1)\nW2 = np.random.randn(n_hidden2, n_hidden1)\nb2 = np.random.randn(n_hidden2)\nW3 = np.random.randn(n_output, n_hidden2)\nb3 = np.random.randn(n_output)\n\nprint(f'\\nW1 shape: {W1.shape}')  # expected: (8, 4)\nprint(f'b1 shape: {b1.shape}')  # expected: (8,)\nprint(f'W2 shape: {W2.shape}')  # expected: (4, 8)\nprint(f'W3 shape: {W3.shape}')  # expected: (2, 4)\n# expected total_params: 82" height="260" >}}

**Professor's hints**

- Layer 1: \\(8 \times 4 = 32\\) weights + 8 biases = 40 parameters.
- Layer 2: \\(4 \times 8 = 32\\) weights + 4 biases = 36 parameters.
- Layer 3: \\(2 \times 4 = 8\\) weights + 2 biases = 10 parameters.
- Total: 40 + 36 + 10 = **86 parameters**. (Note: n_hidden1=8, n_hidden2=4, n_output=2 → recalculate.)
- Weight matrix shape for layer \\(\ell\\): `(n_out, n_in)` — rows are output neurons, columns are input neurons.

**Common pitfalls**

- **Confusing weight matrix orientation:** \\(W \in \mathbb{R}^{n_{out} \times n_{in}}\\) so the operation is \\(Wx\\) with \\(x \in \mathbb{R}^{n_{in}}\\). If you use \\(W \in \mathbb{R}^{n_{in} \times n_{out}}\\) you need to transpose: \\(W^T x\\). Pick one convention and stick to it.
- **Forgetting bias parameters:** Each layer has both a weight matrix AND a bias vector. Don't forget to count the biases.
- **Using the wrong shape for np.random.randn:** `np.random.randn(n_out, n_in)` creates a matrix, `np.random.randn(n_out)` creates a vector. Use the correct form for each.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np
np.random.seed(42)

n_input, n_hidden1, n_hidden2, n_output = 4, 8, 4, 2

# Parameter counts
params_layer1 = n_hidden1 * n_input + n_hidden1   # 8*4 + 8 = 40
params_layer2 = n_hidden2 * n_hidden1 + n_hidden2  # 4*8 + 4 = 36
params_layer3 = n_output * n_hidden2 + n_output    # 2*4 + 2 = 10
total_params  = params_layer1 + params_layer2 + params_layer3  # 86

print(f'Total params: {total_params}')  # 86

W1 = np.random.randn(n_hidden1, n_input)   # (8, 4)
b1 = np.random.randn(n_hidden1)             # (8,)
W2 = np.random.randn(n_hidden2, n_hidden1) # (4, 8)
b2 = np.random.randn(n_hidden2)             # (4,)
W3 = np.random.randn(n_output, n_hidden2)  # (2, 4)
b3 = np.random.randn(n_output)              # (2,)
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** A network solving XOR. Use these pre-given weights to do a forward pass on all 4 XOR inputs and verify the correct outputs.

{{< pyrepl code="import numpy as np\n\n# Pre-given weights that solve XOR\n# Hidden layer: 2 neurons, ReLU\nW1 = np.array([[1.0, 1.0], [-1.0, -1.0]])\nb1 = np.array([-0.5, 1.5])\n# Output layer: 1 neuron, step function\nW2 = np.array([[1.0, -1.0]])\nb2 = np.array([-0.5])\n\nXOR_inputs  = np.array([[0,0],[0,1],[1,0],[1,1]])\nXOR_outputs = np.array([0, 1, 1, 0])\n\nfor x, y_true in zip(XOR_inputs, XOR_outputs):\n    # Forward pass\n    h = np.maximum(0, W1 @ x + b1)      # ReLU hidden layer\n    out = W2 @ h + b2                    # linear output\n    y_hat = 1 if out[0] > 0 else 0\n    print(f'x={x}, y_true={y_true}, y_hat={y_hat}')\n# expected: all y_hat match y_true" height="220" >}}

2. **Coding:** Calculate the parameter count for the DQN network used in Atari: input=84×84×4 pixels (=28224 features), hidden1=512, hidden2=256, output=18 actions. How many parameters does this have?
3. **Challenge:** The universal approximation theorem states that a single-hidden-layer MLP can approximate any continuous function to arbitrary precision. But in practice, deeper networks are used instead. Why? What are the practical advantages of depth over width?
4. **Variant:** A 5-layer MLP with layers [2, 4, 8, 4, 2, 1]. Count the total parameters. Initialize all weights as NumPy arrays and print shapes.
5. **Debug:** The MLP below has a transposed weight matrix in the second layer, causing a shape mismatch. Find and fix the bug.

{{< pyrepl code="import numpy as np\nnp.random.seed(0)\n\nx = np.array([0.5, -0.3, 0.8])\n\nW1 = np.random.randn(4, 3)   # (4 out, 3 in) — correct\nb1 = np.random.randn(4)\nW2 = np.random.randn(4, 2)   # BUG: shape is wrong — should be (2, 4)\nb2 = np.random.randn(2)\n\nh1 = np.maximum(0, W1 @ x + b1)   # shape (4,) — correct\n# TODO: fix the matrix dimensions so this works\nout = W2 @ h1 + b2                 # will fail with current W2 shape\nprint('output:', out)\n# expected: output shape (2,)" height="200" >}}

6. **Conceptual:** Why do we need non-linear activations between layers? Show algebraically that two consecutive linear layers without activation collapse to a single linear layer. Let \\(h = W_2(W_1 x + b_1) + b_2\\) and simplify.
7. **Recall:** Name the three types of layers in an MLP and what computation each performs. State the weight matrix shape convention (rows = ?, columns = ?).
