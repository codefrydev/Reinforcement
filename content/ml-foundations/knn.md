---
title: "K-Nearest Neighbors"
description: "Classify new points by majority vote among K closest training examples."
date: 2026-03-20T00:00:00Z
weight: 10
draft: false
difficulty: 4
tags: ["KNN", "k-nearest neighbors", "distance", "classification", "ml-foundations"]
keywords: ["k-nearest neighbors", "euclidean distance", "lazy learning", "classification", "KNN from scratch"]
roadmap_icon: "database"
roadmap_color: "teal"
roadmap_phase_label: "Chapter 10"
---

**Learning objectives**

- Explain the KNN algorithm and why it is called "lazy learning."
- Compute Euclidean distance between two points by hand and in NumPy.
- Implement KNN classification from scratch and observe the effect of varying K.

**Concept and real-world motivation**

To classify a new point, look at the K closest points in the training set and take a majority vote. There is no explicit training phase — the model is literally just the stored training data. This makes KNN a **lazy learner**: all computation happens at prediction time. It works surprisingly well when: (1) similar inputs have similar outputs, and (2) the dataset is not too large.

KNN naturally handles non-linear decision boundaries because it adapts locally to the data. The critical hyperparameter is K: K=1 memorizes every training point (high variance, low bias), large K smooths the boundary (low variance, high bias). In RL, **memory-based methods** like episodic Q-learning store past experience and look up similar states at decision time — the same lazy-learning idea applied to value functions.

**Illustration:** KNN accuracy peaks at a middle value of K, then degrades as K grows too large.

{{< chart type="bar" palette="learning" title="KNN Accuracy vs K" labels="K=1,K=3,K=5,K=7,K=9,K=11" data="0.72,0.85,0.88,0.87,0.84,0.82" xLabel="K" yLabel="Accuracy" >}}

**Exercise:** Implement KNN classification from scratch on 8 training points in 2D. Classify one test point using K=3.

{{< pyrepl code="import numpy as np\n\n# Training data: 8 points (x, y) with labels 0 or 1\nX_train = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2]], dtype=float)\ny_train = np.array([0, 0, 1, 1, 0, 1, 1, 1])\n\ntest_point = np.array([5, 5], dtype=float)\nK = 3\n\n# TODO: compute Euclidean distance from test_point to each training point\n# Hint: np.sqrt(np.sum((X_train - test_point)**2, axis=1))\ndistances = None\n\n# TODO: get indices that would sort distances (ascending)\nsorted_idx = None  # np.argsort(distances)\n\n# TODO: take first K indices, get their labels, majority vote\nk_nearest_labels = None\nprediction = None  # int(np.mean(k_nearest_labels) >= 0.5)\n\nprint(f'Distances: {distances}')\nprint(f'K={K} nearest labels: {k_nearest_labels}')\nprint(f'Predicted class: {prediction}')\n# expected: prediction = 1 (the 3 nearest points are mostly class 1)" height="280" >}}

**Professor's hints**

- Euclidean distance: \\(d(p,q) = \sqrt{\sum_i (p_i - q_i)^2}\\). In NumPy: `np.sqrt(np.sum((X_train - test_point)**2, axis=1))` computes all 8 distances in one line.
- `np.argsort(distances)` returns indices that sort the distances from smallest to largest. Take the first K of those.
- Majority vote: `int(np.mean(k_nearest_labels) >= 0.5)` works for binary labels. More generally, use `np.bincount(k_nearest_labels).argmax()`.

**Common pitfalls**

- **Forgetting to normalize features:** If one feature ranges 0–1000 and another 0–1, distance is dominated by the first. Always scale features before KNN.
- **Using wrong axis in argsort:** `np.argsort(distances)` sorts a 1D array correctly. If you accidentally keep `axis=1`, you sort within rows, not across all distances.
- **K larger than training set:** K cannot exceed the number of training points. Add a check: `K = min(K, len(X_train))`.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np

X_train = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2]], dtype=float)
y_train = np.array([0, 0, 1, 1, 0, 1, 1, 1])
test_point = np.array([5, 5], dtype=float)
K = 3

distances     = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
sorted_idx    = np.argsort(distances)
k_nearest_labels = y_train[sorted_idx[:K]]
prediction    = int(np.mean(k_nearest_labels) >= 0.5)
print(f'Predicted class: {prediction}')  # 1
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Compute the Euclidean distance from point `A=(1,2)` to points `B=(4,6)` and `C=(1,3)` by hand. Which is closer?

{{< pyrepl code="import numpy as np\nA = np.array([1, 2])\nB = np.array([4, 6])\nC = np.array([1, 3])\n# TODO: compute distances d(A,B) and d(A,C)\ndAB = None  # expected: 5.0\ndAC = None  # expected: 1.0\nprint(f'd(A,B)={dAB}, d(A,C)={dAC}')" height="180" >}}

2. **Coding:** Wrap the KNN logic into a function `knn_predict(X_train, y_train, test_point, K)` and test it on all 8 training points (classifying each using the remaining 7 as training data — leave-one-out).
3. **Challenge:** Generate a 2D dataset with 100 points from two Gaussian clusters. Run KNN with K=1, 5, and 10 and compute test accuracy for each. Plot how accuracy changes with K.
4. **Variant:** Try K=1, K=5, K=10 on the 8-point training set from the main exercise with test point `(2, 2)`. Record how the predicted label changes. Explain why K=1 is most sensitive to noise.
5. **Debug:** The code below has a bug — `argsort` result is not sliced correctly so wrong neighbors are selected. Find and fix it.

{{< pyrepl code="import numpy as np\nX_train = np.array([[1,2],[5,8],[8,8],[1,0.6],[9,11]], dtype=float)\ny_train = np.array([0, 1, 1, 0, 1])\ntest = np.array([2, 2], dtype=float)\nK = 2\ndists = np.sqrt(np.sum((X_train - test)**2, axis=1))\n# BUG: takes last K instead of first K (farthest instead of nearest)\nk_idx = np.argsort(dists)[-K:]   # BUG\nlabels = y_train[k_idx]\nprint(f'Neighbors (buggy): {k_idx}, labels: {labels}')\n# TODO: fix to get the K NEAREST (smallest distance) neighbors\n# expected: nearest 2 are index 0 and 3 (both class 0)" height="200" >}}

6. **Conceptual:** Why does KNN get slower as the training set grows? What is the time complexity of one prediction given N training points and D features?
7. **Recall:** From memory, describe the KNN prediction algorithm in 4 steps, starting from "given a new test point."
