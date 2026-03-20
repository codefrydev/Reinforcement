---
title: "CNN Basics: Convolutions and Pooling"
description: "Learn convolution and pooling from scratch in NumPy. See how Atari DQN uses CNNs to process raw pixels."
date: 2026-03-20T00:00:00Z
weight: 11
draft: false
difficulty: 5
tags: ["CNN", "convolution", "pooling", "image processing", "DQN", "dl-foundations"]
keywords: ["convolutional neural network", "2D convolution", "max pooling", "Atari DQN", "feature maps", "edge detection"]
roadmap_icon: "network"
roadmap_color: "green"
roadmap_phase_label: "Chapter 11"
---

**Learning objectives**
- Implement 2D convolution from scratch in NumPy
- Implement 2×2 max pooling
- Explain how CNNs extract spatial features from images
- Describe how Atari DQN uses a 3-layer CNN to process raw pixel observations

**Concept and real-world motivation**

A **Convolutional Neural Network (CNN)** learns spatial features from images using **filters** (small weight matrices). Each filter slides across the image and produces a **feature map** — a 2D response showing where that pattern appears. Early filters detect edges; deeper filters combine edges into textures, shapes, and objects.

**Pooling** reduces the spatial size of feature maps, making the network less sensitive to small shifts in position (translation invariance) and reducing computation.

**In RL:** Atari DQN (Mnih et al., 2015) uses **3 convolutional layers** to process stacked 84×84 grayscale frames before the fully-connected Q-value head. The CNN is the **state encoder** — it transforms raw pixels into a compact vector that the Q-network can reason about. Without the CNN, the Q-network would need to process 84×84×4 = 28,224 input features directly.

**Math:**

2D convolution (valid padding): \\((I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m,n]\\)

For an \\(H \times W\\) input and \\(k \times k\\) kernel, the output size is \\((H-k+1) \times (W-k+1)\\).

Max pooling: divide the feature map into non-overlapping regions; take the maximum value from each region.

**Illustration — Atari DQN CNN architecture:**

```
Input Image (84×84×4 stacked frames)
       ↓
Conv Layer 1 (32 filters, 8×8, stride 4) → ReLU
       ↓
Conv Layer 2 (64 filters, 4×4, stride 2) → ReLU
       ↓
Conv Layer 3 (64 filters, 3×3, stride 1) → ReLU
       ↓
Flatten → 3136 features
       ↓
FC Layer (512 units) → ReLU
       ↓
Q-values (one per action)
```

**Exercise:** Implement 2D convolution and max pooling from scratch in NumPy.

{{< pyrepl code="import numpy as np\n\n# 5x5 input image\nimage = np.arange(25).reshape(5, 5).astype(float)\nprint('Input image:')\nprint(image)\n\n# 3x3 edge-detection filter (Laplacian)\nkernel = np.array([[-1,-1,-1],\n                   [-1, 8,-1],\n                   [-1,-1,-1]], dtype=float)\n\n# 2D convolution (valid padding: no padding, stride=1)\ndef conv2d_valid(img, k):\n    H, W = img.shape\n    kH, kW = k.shape\n    out_H = H - kH + 1\n    out_W = W - kW + 1\n    output = np.zeros((out_H, out_W))\n    for i in range(out_H):\n        for j in range(out_W):\n            output[i, j] = np.sum(img[i:i+kH, j:j+kW] * k)\n    return output\n\n# 2x2 max pooling\ndef max_pool2d(feature_map, pool_size=2):\n    H, W = feature_map.shape\n    out_H = H // pool_size\n    out_W = W // pool_size\n    output = np.zeros((out_H, out_W))\n    for i in range(out_H):\n        for j in range(out_W):\n            region = feature_map[i*pool_size:(i+1)*pool_size,\n                                 j*pool_size:(j+1)*pool_size]\n            output[i, j] = region.max()\n    return output\n\nfeature_map = conv2d_valid(image, kernel)\nprint('\\nConv output (3x3):')\nprint(feature_map)\nprint('\\nMax pool output (1x1 from 3x3):')\nprint(max_pool2d(feature_map))" height="300" >}}

**Professor's hints**
- The valid convolution output size is always smaller than the input. Use **same padding** (zero-pad the input) if you want the output to have the same size.
- The Laplacian filter `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` detects regions of rapid change (edges). Positive response means the center is brighter than its surroundings.
- In practice, CNNs learn the filter values by backpropagation — we don't hand-design them.
- Each filter produces one **channel** in the output. Stacking many filters = many channels = richer representations.

**Common pitfalls**
- Off-by-one errors in the output size: remember it's `H - k + 1` for valid conv.
- Confusing "stride" (step size when sliding the filter) with "dilation" (spacing between filter elements).
- Applying max pooling before activation functions instead of after.

{{< collapse summary="Worked solution" >}}
Convolution by hand on the center patch of the 5×5 image:
- Patch: rows 1-3, cols 1-3 of `arange(25).reshape(5,5)` = `[[6,7,8],[11,12,13],[16,17,18]]`
- Laplacian response: 8×12 - (6+7+8+11+13+16+17+18) = 96 - 96 = 0 (center is average of surroundings → no edge)

For max pooling on a 3×3 output, we can only pool with size 2 on the top-left 2×2 region since 3//2=1.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Apply the edge-detection kernel to one 3×3 patch by hand. Use the top-left 3×3 region of the 5×5 image (rows 0-2, cols 0-2).
{{< pyrepl code="import numpy as np\nimage = np.arange(25).reshape(5, 5).astype(float)\nkernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)\n# TODO: extract top-left 3x3 patch and compute dot product with kernel\npatch = image[0:3, 0:3]\nresponse = np.sum(patch * kernel)\nprint('patch:')\nprint(patch)\nprint('convolution response:', response)" height="200" >}}

2. **Coding:** Extend `conv2d_valid` to support a configurable stride. For stride=2, output size is `(H - k) // stride + 1`. This is what Atari DQN uses in its first two conv layers.

3. **Challenge:** Implement a multi-channel convolution: input has 3 channels (like RGB), filter has shape (3, k, k), and the output is the sum of convolutions across input channels. This is how real CNNs process color images.

4. **Variant:** Implement average pooling and compare to max pooling on the edge-detection feature map above. When might average pooling be preferable?

5. **Debug:** Fix the convolution with wrong loop bounds (off by 1):
{{< pyrepl code="import numpy as np\nimg = np.arange(16).reshape(4,4).astype(float)\nk = np.ones((2,2)) / 4  # 2x2 averaging filter\n# BUG: loop goes one step too far\ndef conv_buggy(img, k):\n    H, W = img.shape\n    kH, kW = k.shape\n    out = np.zeros((H - kH + 1, W - kW + 1))\n    for i in range(H - kH + 2):  # BUG: +2 should be +1\n        for j in range(W - kW + 2):  # BUG: +2 should be +1\n            out[i, j] = np.sum(img[i:i+kH, j:j+kW] * k)\n    return out\nprint('buggy output shape:', conv_buggy(img, k).shape, '  expected: (3, 3)')\n# TODO: fix the loop bounds" height="220" >}}

6. **Notebook:** For a full PyTorch CNN implementation, use the local notebook:

{{< notebook path="dl-foundations/cnn_pytorch.ipynb" title="CNN in PyTorch (run locally)" >}}

7. **Recall:** In your own words: (a) What does a convolutional filter detect? (b) Why does DQN need a CNN instead of a plain MLP for Atari? (c) What is the effect of increasing the filter size (e.g. 3×3 → 5×5)?
