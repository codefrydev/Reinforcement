---
title: "K-Means Clustering"
description: "Unsupervised grouping of data by alternating assignment and centroid-update steps."
date: 2026-03-20T00:00:00Z
weight: 12
draft: false
difficulty: 4
tags: ["clustering", "k-means", "unsupervised learning", "centroids", "ml-foundations"]
keywords: ["k-means clustering", "unsupervised learning", "centroids", "assignment step", "convergence"]
roadmap_icon: "network"
roadmap_color: "amber"
roadmap_phase_label: "Chapter 12"
---

**Learning objectives**

- Explain why K-Means is unsupervised and when it is appropriate.
- Execute the assignment and update steps of K-Means by hand on a small dataset.
- Identify convergence and understand the role of initialization.

**Concept and real-world motivation**

All the algorithms so far required labels. **Unsupervised learning** finds structure in data with no labels at all. **K-Means** groups unlabeled points into K clusters by alternating between two steps: (1) **assignment** — assign each point to the nearest centroid; (2) **update** — move each centroid to the mean of its assigned points. Repeating until assignments stop changing guarantees convergence (though not necessarily to the global optimum).

K-Means has countless applications: customer segmentation, image compression (quantize pixel colors), document clustering, and anomaly detection. In RL, we sometimes cluster similar states together to build a smaller **abstract state space** — reducing the problem size before training a policy. This is called state abstraction or tile coding.

**Illustration:** Final cluster sizes after K-Means converges on a 3-cluster dataset.

{{< chart type="bar" palette="math" title="Cluster Sizes at Convergence" labels="Cluster 1,Cluster 2,Cluster 3" data="35,28,37" xLabel="Cluster" yLabel="Points" >}}

**Exercise:** Run 3 iterations of K-Means by hand on 6 points in 2D.

{{< pyrepl code="import numpy as np\n\n# 6 points clearly forming 2 natural clusters\npoints = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]], dtype=float)\n\n# Initial centroids\ncentroids = np.array([[1,1],[8,8],[5,5]], dtype=float)\nK = 3\n\nfor iteration in range(3):\n    # TODO: Assignment step — assign each point to nearest centroid\n    # Hint: for each point, compute distance to each centroid, take argmin\n    assignments = []\n    for p in points:\n        dists = None  # np.linalg.norm(centroids - p, axis=1)\n        cluster = None  # np.argmin(dists)\n        assignments.append(cluster)\n    assignments = np.array(assignments)\n\n    # TODO: Update step — new centroid = mean of assigned points\n    new_centroids = np.zeros_like(centroids)\n    for k in range(K):\n        mask = assignments == k\n        if mask.sum() > 0:\n            new_centroids[k] = None  # points[mask].mean(axis=0)\n        else:\n            new_centroids[k] = centroids[k]  # keep if empty\n\n    print(f'Iteration {iteration+1}: assignments={assignments}')\n    print(f'  New centroids: {new_centroids.tolist()}')\n    centroids = new_centroids\n\n# expected: after 2-3 iterations, cluster 0 → points 0-2, cluster 1 → points 3-5" height="300" >}}

**Professor's hints**

- `np.linalg.norm(centroids - p, axis=1)` computes the distance from point `p` to each of the K centroids in one line. `axis=1` sums across the 2 feature dimensions.
- The update step is simply the mean: `points[mask].mean(axis=0)`. If a cluster is empty (no points assigned), keep the old centroid or re-initialize — never divide by zero.
- Convergence: when `np.array_equal(assignments, prev_assignments)`. After iteration 2, the assignments in this example should be stable.

**Common pitfalls**

- **Dividing by zero in update:** If a cluster gets no assigned points, `mean([]) = NaN`. Always check `mask.sum() > 0`.
- **Sensitivity to initialization:** K-Means can converge to different local optima depending on initial centroids. Run multiple times with different seeds (sklearn's `n_init` parameter) and keep the best result (lowest inertia).
- **Choosing K:** K is a hyperparameter. Use the elbow method — plot inertia vs K — and look for the "elbow" where adding more clusters gives diminishing returns.

{{< collapse summary="Worked solution" >}}
```python
import numpy as np
points    = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]], dtype=float)
centroids = np.array([[1,1],[8,8],[5,5]], dtype=float)

for it in range(3):
    dists       = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    new_c = np.zeros_like(centroids)
    for k in range(3):
        mask = assignments == k
        new_c[k] = points[mask].mean(axis=0) if mask.sum() > 0 else centroids[k]
    print(f'it{it+1}: assign={assignments}')
    centroids = new_c
```
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** Plot the 6 points from the main exercise colored by their cluster assignment after the final iteration. Use a simple ASCII-style print or matplotlib scatter.

{{< pyrepl code="import numpy as np\nimport matplotlib.pyplot as plt\n\npoints = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]], dtype=float)\nassignments = np.array([0, 0, 0, 1, 1, 1])  # after convergence\n\ncolors = ['blue', 'red', 'green']\n# TODO: create a scatter plot with points colored by cluster\nfor k in range(2):\n    mask = assignments == k\n    plt.scatter(points[mask, 0], points[mask, 1], c=colors[k], label=f'Cluster {k}')\nplt.legend()\nplt.title('K-Means Clusters')\nplt.show()\n# expected: two distinct colored groups in the plot" height="220" >}}

2. **Coding:** Wrap K-Means into a function `kmeans(points, K, n_iter=10)` that returns final assignments and centroids. Test it with K=2 and K=3 on the 6-point dataset.
3. **Challenge:** Generate 100 2D points from 3 Gaussian clusters using `np.random.randn`. Run K-Means with K=2, K=3, K=4. Compute **inertia** (sum of squared distances from each point to its centroid) for each K. Plot inertia vs K and identify the elbow.
4. **Variant:** Try 3 different initial centroid choices on the 6-point dataset: (a) the ones in the exercise, (b) three random points, (c) the first three data points. Do all initializations converge to the same clusters?
5. **Debug:** The code below has a bug — the centroid update divides by the total number of points instead of the cluster size. Find and fix it.

{{< pyrepl code="import numpy as np\npoints = np.array([[1,1],[1,2],[2,1],[8,8],[8,9],[9,8]], dtype=float)\nassignments = np.array([0,0,0,1,1,1])\nK = 2\n\nnew_centroids = np.zeros((K, 2))\nfor k in range(K):\n    mask = assignments == k\n    # BUG: divides by total N instead of cluster size\n    new_centroids[k] = points[mask].sum(axis=0) / len(points)\nprint('Buggy centroids:', new_centroids)\n# TODO: fix to divide by the number of points in cluster k\n# expected: centroid 0 = [1.33, 1.33], centroid 1 = [8.33, 8.33]" height="200" >}}

6. **Conceptual:** K-Means minimizes within-cluster sum of squared distances (inertia). Is this guaranteed to find the global minimum? Why or why not?
7. **Recall:** From memory, write the two steps of the K-Means algorithm (assignment and update) and state what "convergence" means.
