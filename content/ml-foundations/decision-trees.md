---
title: "Decision Trees"
description: "If/else questions on features, entropy, and information gain as splitting criteria."
date: 2026-03-20T00:00:00Z
weight: 11
draft: false
difficulty: 4
tags: ["decision trees", "entropy", "information gain", "classification", "ml-foundations"]
keywords: ["decision tree", "entropy", "information gain", "splitting criteria", "interpretable model"]
roadmap_icon: "layers"
roadmap_color: "green"
roadmap_phase_label: "Chapter 11"
---

**Learning objectives**

- Describe how a decision tree splits data using if/else thresholds on features.
- Compute entropy \\(H(S)\\) for a dataset with binary labels.
- Calculate information gain for a proposed split and choose the best splitting threshold.

**Concept and real-world motivation**

A decision tree answers the question "which class does this belong to?" through a sequence of if/else questions. At each internal node, the tree picks a feature and a threshold (e.g. "is feature1 > 3?"). Points that satisfy the condition go left; the rest go right. This repeats until reaching leaf nodes that assign class labels. The tree learns by choosing splits that **maximize information gain** — the reduction in uncertainty (entropy) after the split.

Decision trees are appealing because they are interpretable: you can trace exactly why a prediction was made. They also handle non-linear boundaries naturally. In RL, decision trees appear as interpretable policy representations: instead of a black-box neural network, an agent's policy can be a tree — "if speed > 5 and obstacle_ahead, then brake." This allows human inspection of learned behaviors.

**Illustration:** A two-level decision tree splitting on feature1, then feature2.

{{< mermaid >}}
graph TD
    A["feature1 > 3?"] -->|Yes| B["feature2 > 1?"]
    A -->|No| C["Class A"]
    B -->|Yes| D["Class B"]
    B -->|No| E["Class A"]
{{< /mermaid >}}

**Exercise:** For a dataset with feature `x = [1,2,3,4,5,6]` and `labels = [0,0,0,1,1,1]`:
1. Compute entropy of the full dataset.
2. Try split at `x = 3.5`: compute entropy of each side.
3. Compute information gain for this split.

{{< pyrepl code="import numpy as np\n\nlabels = np.array([0, 0, 0, 1, 1, 1])\n\ndef entropy(y):\n    # TODO: H(y) = -sum(p_c * log2(p_c)) for each class c\n    # Hint: use np.bincount and np.log2\n    # Handle p=0 case: 0 * log2(0) = 0\n    n = len(y)\n    if n == 0:\n        return 0.0\n    counts = np.bincount(y)\n    probs = counts[counts > 0] / n\n    return None  # TODO: -np.sum(probs * np.log2(probs))\n\n# Split at x = 3.5: left = labels[0:3], right = labels[3:6]\nleft  = labels[:3]   # [0, 0, 0]\nright = labels[3:]   # [1, 1, 1]\n\nH_parent = entropy(labels)\nH_left   = entropy(left)\nH_right  = entropy(right)\n\nn = len(labels)\nweighted_H = (len(left)/n) * H_left + (len(right)/n) * H_right\ninfo_gain  = H_parent - weighted_H\n\nprint(f'H(parent) = {H_parent:.4f}')\nprint(f'H(left)   = {H_left:.4f},  H(right) = {H_right:.4f}')\nprint(f'Weighted H = {weighted_H:.4f}')\nprint(f'Information gain = {info_gain:.4f}')\n# expected: H(parent)=1.0 bit, H(left)=0, H(right)=0, gain=1.0 bit" height="260" >}}

**Professor's hints**

- Entropy formula: \\(H(S) = -\sum_c p_c \log_2 p_c\\). Use `np.bincount(y)` to count each class, divide by `len(y)` for probabilities.
- Pure node (all one class) has entropy 0. Perfectly mixed node (50/50 binary) has entropy 1 bit. These are the extremes.
- Information gain = \\(H(\text{parent}) - \frac{|L|}{N} H(L) - \frac{|R|}{N} H(R)\\). Maximum gain means the split perfectly separates classes.

**Common pitfalls**

- **Using `log` instead of `log2`:** Entropy in bits uses \\(\log_2\\). Using natural log gives entropy in nats — numerically different. Both are valid measures, but be consistent.
- **Zero probability terms:** If a class is absent from a split, \\(p = 0\\) and \\(0 \times \log_2(0)\\) is undefined mathematically but defined as 0 by convention. Filter out zero-count classes with `counts[counts > 0]`.
- **Overfitting to full depth:** A tree grown until every leaf is pure will memorize training data. Always use `max_depth` or `min_samples_leaf` to regularize.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / n
    return -np.sum(probs * np.log2(probs))

labels = np.array([0, 0, 0, 1, 1, 1])
left, right = labels[:3], labels[3:]

H_parent   = entropy(labels)         # 1.0 bit (perfectly mixed)
H_left     = entropy(left)           # 0.0 (all zeros)
H_right    = entropy(right)          # 0.0 (all ones)
n = len(labels)
weighted_H = (len(left)/n)*H_left + (len(right)/n)*H_right  # 0.0
info_gain  = H_parent - weighted_H   # 1.0 bit (perfect split!)
print(f'Information gain = {info_gain:.4f}')
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Compute entropy by hand for a dataset with 4 positive and 2 negative examples. \\(H = -\frac{4}{6}\log_2\frac{4}{6} - \frac{2}{6}\log_2\frac{2}{6}\\).

{{< pyrepl code="import numpy as np\n# 4 positives, 2 negatives → total 6\n# TODO: compute H([4 pos, 2 neg])\np_pos = 4/6\np_neg = 2/6\nH = None  # -p_pos*np.log2(p_pos) - p_neg*np.log2(p_neg)\nprint(f'H = {H:.4f} bits')\n# expected: ≈ 0.9183 bits" height="180" >}}

2. **Coding:** Write a function `best_split(X, y)` that tries all possible thresholds (midpoints between consecutive sorted values) and returns the feature and threshold with maximum information gain.
3. **Challenge:** Implement a full decision tree (just depth-2) from scratch: at the root, find the best split; at each child, find the best split again; leaves predict majority class. Test on the 6-sample dataset from the exercise.
4. **Variant:** Change the labels to `[0,0,1,0,1,1]` (not cleanly separable at x=3.5). Recompute information gain for the same split. How does it compare to the 1.0-bit gain in the main exercise?
5. **Debug:** The code below has a bug — it uses `np.log` (natural log) instead of `np.log2`, making entropy values numerically wrong. Find and fix it.

{{< pyrepl code="import numpy as np\n\ndef buggy_entropy(y):\n    n = len(y)\n    counts = np.bincount(y)\n    probs = counts[counts > 0] / n\n    return -np.sum(probs * np.log(probs))  # BUG: should be log2\n\ny = np.array([0, 0, 1, 1])\nprint(f'Buggy H = {buggy_entropy(y):.4f}')  # prints ~0.693 (nats)\n# TODO: fix so it returns 1.0 (bits for 50/50 split)\n# expected: H([0,0,1,1]) = 1.0 bit" height="200" >}}

6. **Conceptual:** Why is a decision tree prone to overfitting when grown to full depth? Name two hyperparameters you can set in sklearn's `DecisionTreeClassifier` to control tree size.
7. **Recall:** From memory, write the entropy formula \\(H(S)\\) and the information gain formula in terms of parent and child entropies.
