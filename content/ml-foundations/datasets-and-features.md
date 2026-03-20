---
title: "Datasets and Features"
description: "Features, labels, and how ML data is structured as rows (samples) and columns (features) in a DataFrame."
date: 2026-03-20T00:00:00Z
weight: 2
draft: false
difficulty: 4
tags: ["datasets", "features", "labels", "pandas", "data loading", "ml-foundations"]
keywords: ["features and labels", "X and y", "DataFrame", "pandas", "train test split", "ML data structure"]
roadmap_icon: "database"
roadmap_color: "teal"
roadmap_phase_label: "Chapter 2"
---

**Learning objectives**

- Define features (X) and labels (y) and explain the role of each in supervised learning.
- Load and inspect a dataset using pandas: `.head()`, `.describe()`, `.shape`, `.dtypes`.
- Split a DataFrame into a feature matrix X and a label vector y.

**Concept and real-world motivation**

Every supervised learning problem has the same shape: a table of **examples** where each row is one observation and each column is one **feature** (an input measurement). One special column is the **label** — the thing we want to predict. The feature matrix is called **X** (capital, because it is a matrix) and the label vector is called **y** (lower-case, because it is a vector). When you train a model, you show it (X, y) pairs so it can learn the function \\(f: X \to y\\).

In practice, data arrives as CSV files, databases, or API responses. The `pandas` library gives us a `DataFrame` — a table with named columns — which is the standard container for ML data in Python. Before touching any model, a practitioner always inspects the data: How many samples? How many features? Any missing values? What are the value ranges? This step is called **exploratory data analysis (EDA)**, and skipping it is the single most common source of silent bugs in ML pipelines.

**RL connection:** In RL, the agent observes a **state** at each timestep. That state is a vector of numbers — position, velocity, sensor readings, pixel values. State observations *are* features. When we later approximate the value function as \\(V(s) \approx w^T s\\), we are treating the state exactly like an X matrix from supervised learning. Every trick you learn here for handling features applies directly to state representations in RL.

**Illustration:** The diagram below shows how a DataFrame maps onto the supervised learning framework.

{{< mermaid >}}
flowchart LR
    subgraph DataFrame
        direction TB
        F["Features (X)\nheight | weight | age"]
        L["Label (y)\nhealthy"]
    end
    F -->|"model f(X)"| P["Prediction ŷ"]
    L -->|"compare"| P
{{< /mermaid >}}

Here is what a small health dataset looks like in pandas — run this to inspect it:

{{< pyrepl code="import pandas as pd\n\ndata = {\n    'height': [165, 178, 155, 190, 172],\n    'weight': [60, 80, 55, 95, 70],\n    'age':    [25, 32, 28, 45, 38],\n    'healthy': [1, 1, 1, 0, 1]\n}\ndf = pd.DataFrame(data)\n\nprint('Shape:', df.shape)        # (rows, columns)\nprint()\nprint(df.head())                  # first few rows\nprint()\nprint(df.describe())              # summary statistics" height="260" >}}

**Exercise:** Using the same dataset, split it into features X and label y. Then explore X with `.shape`, `.columns`, and `.values`. Finally, check the label distribution with `y.value_counts()`.

{{< pyrepl code="import pandas as pd\n\ndata = {\n    'height': [165, 178, 155, 190, 172],\n    'weight': [60, 80, 55, 95, 70],\n    'age':    [25, 32, 28, 45, 38],\n    'healthy': [1, 1, 1, 0, 1]\n}\ndf = pd.DataFrame(data)\n\n# TODO: split into X (features) and y (label)\n# X should contain height, weight, age\n# y should contain the healthy column\nX = None  # replace with correct code\ny = None  # replace with correct code\n\n# TODO: print X.shape, X.columns, and y.value_counts()\n# expected: X.shape = (5, 3), X.columns = Index(['height','weight','age'])\n# expected: y has value 1 appearing 4 times, value 0 appearing 1 time" height="260" >}}

**Professor's hints**

- To select multiple columns from a DataFrame: `df[['col1', 'col2', 'col3']]` (double brackets → DataFrame).
- To select a single column as a Series: `df['col']` (single brackets → Series).
- `X = df.drop('healthy', axis=1)` is a clean way to get all columns except the label.
- `y = df['healthy']` gets the label column as a 1-D Series.
- `.shape` returns `(n_samples, n_features)` — always check this first to make sure the split is correct.

**Common pitfalls**

- **Including the label in X:** Always double-check that X does not contain the y column. If the model sees the answer during training, it will appear perfect but fail completely on new data.
- **Forgetting to reset the index:** After filtering rows, pandas row indices may not start at 0. Call `.reset_index(drop=True)` before training to avoid index-related errors.
- **Treating all columns as features:** Columns like "customer ID" or "timestamp" are identifiers, not features. Including them can cause the model to memorize IDs rather than learn patterns.

{{< collapse summary="Worked solution" >}}
Split X and y correctly, then explore:

```python
import pandas as pd

data = {
    'height': [165, 178, 155, 190, 172],
    'weight': [60, 80, 55, 95, 70],
    'age':    [25, 32, 28, 45, 38],
    'healthy': [1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Split
X = df[['height', 'weight', 'age']]   # or: df.drop('healthy', axis=1)
y = df['healthy']

# Explore
print(X.shape)      # (5, 3)
print(X.columns)    # Index(['height', 'weight', 'age'])
print(y.value_counts())
# 1    4
# 0    1
```

Key takeaway: `X` is a (5, 3) matrix — 5 samples, 3 features. `y` is a length-5 vector.
{{< /collapse >}}

**Extra practice**

1. **Warm-up:** A dataset has columns: `[user_id, age, income, city, clicked_ad]`. Which columns should go in X? Which in y? Which should you drop entirely and why?
2. **Coding:** Create a new DataFrame with 8 samples and 4 features of your choice. Add a binary label column. Split into X and y, then print X.describe() to see feature statistics.
3. **Challenge:** Some datasets have **missing values** (NaN). Add a `None` value to the weight column in the example dataset. Then use `df.isnull().sum()` to detect it, and `df.fillna(df.mean())` to fill it. How does this affect `.describe()`?
4. **Variant:** Use `df.dtypes` to see the data type of each column. What happens if the 'healthy' column is stored as a string `'yes'`/`'no'` instead of 1/0? How would you convert it?
5. **Debug:** The code below swaps X and y. Find and fix the bug.

{{< pyrepl code="import pandas as pd\n\ndata = {'height': [165, 178, 155], 'weight': [60, 80, 55], 'healthy': [1, 1, 0]}\ndf = pd.DataFrame(data)\n\n# BUG: X and y are swapped\ny = df[['height', 'weight']]   # BUG: this should be X\nX = df['healthy']              # BUG: this should be y\n\nprint('X shape:', X.shape)   # expected: (3, 2) but will print (3,)\nprint('y shape:', y.shape)   # expected: (3,) but will print (3, 2)\n\n# TODO: fix the assignment of X and y" height="220" >}}

6. **Conceptual:** In a supervised learning problem, what is the difference between a **sample** and a **feature**? Give an example where confusing the two would cause a training error.
7. **Recall:** State the convention for naming the feature matrix and label vector in Python ML (what letters are used and why they are capitalized the way they are).
