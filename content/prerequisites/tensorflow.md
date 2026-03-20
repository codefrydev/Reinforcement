---
title: "TensorFlow"
description: "TensorFlow and Keras for RL: models, GradientTape, optimizers, and GPU."
date: 2026-03-10T00:00:00Z
weight: 60
draft: false
difficulty: 6
tags: ["TensorFlow", "Keras", "GradientTape", "RL", "prerequisites"]
keywords: ["TensorFlow for RL", "Keras", "GradientTape", "optimizers", "GPU", "RL models"]
roadmap_icon: "brain"
roadmap_color: "indigo"
roadmap_phase_label: "Phase 6 · TensorFlow"
---

Alternative to PyTorch for implementing DQN, policy gradients, and other deep RL algorithms. The Keras API provides layers and optimizers; `GradientTape` gives full control over custom loss functions (e.g. policy gradient, CQL).

---

## Why TensorFlow matters for RL

- **Keras API** — `tf.keras.Sequential`, `tf.keras.Model`, layers (Dense, Conv2D). Quick prototyping of Q-networks and policies.
- **Gradient tape** — `tf.GradientTape()` records operations so you can compute gradients of any scalar with respect to trainable variables. Essential for policy gradient and custom losses.
- **Optimizers** — `tf.keras.optimizers.Adam`, `apply_gradients`.
- **Device placement** — GPU via `tf.config` when available.

---

## Core concepts with examples

### Dense layers and Sequential model

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2),  # Q-values for 2 actions
])
model.build(input_shape=(None, 4))
```

### Forward pass and MSE loss

```python
states = tf.random.normal((32, 4))
q_values = model(states)
targets = tf.random.normal((32, 2))
loss = tf.reduce_mean((q_values - targets) ** 2)
```

### Training step with GradientTape

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(states, targets):
    with tf.GradientTape() as tape:
        q_values = model(states)
        loss = tf.reduce_mean((q_values - targets) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

loss_val = train_step(states, targets)
```

### Subclassing for custom models

```python
class QNetwork(tf.keras.Model):
    def __init__(self, n_actions=2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(n_actions)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
```

---

## Worked examples

{{< collapse summary="Worked example 1: Q-network in TensorFlow (click to expand)" >}}

Building and training a simple Q-network — input: 4 features, one hidden layer of 16 neurons, output: Q-values for 2 actions. We train it on one batch of (state, target\_Q) pairs.

```python
import tensorflow as tf
import numpy as np
np.random.seed(42)
tf.random.set_seed(42)

# Build Q-network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(2)  # output: Q-values for 2 actions
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

# Fake batch: 5 state-target pairs
states = np.random.randn(5, 4).astype(np.float32)
target_Q = np.random.randn(5, 2).astype(np.float32)

# Training step
loss = model.train_on_batch(states, target_Q)
print(f'Loss after 1 batch: {loss:.4f}')
predictions = model.predict(states[:2])
print(f'Q-values for first 2 states:\n{predictions}')
```

{{< /collapse >}}

{{< collapse summary="Worked example 2: GradientTape policy gradient (click to expand)" >}}

Using `GradientTape` for a custom REINFORCE-style policy gradient step: we compute \\(-\log \pi(a|s) \cdot R\\) and apply gradients manually.

```python
import tensorflow as tf
import numpy as np

# Policy network: state → action probabilities
policy = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(2, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(lr=0.01)

# Simulate one REINFORCE update
state = tf.constant([[0.5, -0.3, 0.8]], dtype=tf.float32)
action_taken = 1  # agent chose action 1
reward = 1.0      # received reward +1

with tf.GradientTape() as tape:
    probs = policy(state)  # [p(a0), p(a1)]
    log_prob = tf.math.log(probs[0, action_taken])
    loss = -log_prob * reward  # REINFORCE: maximize log_prob * reward

grads = tape.gradient(loss, policy.trainable_variables)
optimizer.apply_gradients(zip(grads, policy.trainable_variables))
print(f'Policy gradient loss: {loss.numpy():.4f}')
print(f'Action probabilities: {probs.numpy()}')
```

{{< /collapse >}}

---

## Exercises

**Exercise 1.** Create a `Sequential` model with one hidden layer (64 units, ReLU) and output dimension 2. Build it with `input_shape=(4,)`. Call `model(tf.random.normal((10, 4)))` and print the output shape. Then use `model.summary()` to inspect parameters.

**Exercise 2.** In a `GradientTape` scope, compute \\(y = x^2\\) for \\(x = tf.constant(3.0)\\) and then `tape.gradient(y, x)`. Verify the gradient is 6.0. (Use a variable: `x = tf.Variable(3.0)` so it's differentiable.)

**Exercise 3.** Implement a training step that: (1) takes `states` (32, 4) and `targets` (32, 2); (2) inside `GradientTape`, computes Q-values from your model and MSE loss; (3) computes gradients and applies them with an Adam optimizer. Run 50 steps with random data and print the loss every 10 steps.

**Exercise 4.** Implement a softmax policy: a small model that maps state (4,) to logits (2,). Given a batch of states, compute action probabilities with `tf.nn.softmax(logits)`. Sample actions with `tf.random.categorical(tf.math.log(probs), 1)`. Return both the sampled actions and the log-probabilities of those actions (using `tf.math.log` and gather).

**Exercise 5.** Create a subclassed `tf.keras.Model` with two dense layers (64, ReLU) and output 2. Override `call(self, inputs)`. Train it for 100 steps with random states and targets using `GradientTape` and Adam. Store the loss in a list and plot it (e.g. with matplotlib) to confirm it decreases.

**Exercise 6.** Create a Variable `x = tf.Variable(2.0)` and inside `GradientTape()` compute `y = x ** 2`, then `grad = tape.gradient(y, x)`. Verify grad is 4.0. **In RL:** GradientTape records ops so policy and value gradients can be computed for custom losses.

**Exercise 7.** Build a small model (4 → 64 → 2). In a loop, generate random (32, 4) states and (32, 2) targets, call your train step, and append the loss to a list. Plot the list with matplotlib. **In RL:** This mirrors the inner loop of DQN or actor-critic training.

**Exercise 8.** (Challenge) Implement a softmax policy that takes state (batch, 4) and returns (actions, log_probs). Use `tf.random.categorical` for sampling. Train with a dummy "loss" = -mean(log_probs) for 50 steps and confirm loss decreases (you are maximizing log-prob). **In RL:** This is the core of REINFORCE-style updates.

---

## Professor's hints

- **In RL:** Use `GradientTape()` for policy gradient and any loss that is not a simple Keras built-in. Record the forward pass inside the tape, then `tape.gradient(loss, model.trainable_variables)`.
- Wrap the training step in `@tf.function` for speed after you have verified it works in eager mode. Be careful: Python side effects (e.g. appending to a list) inside `tf.function` may not run as expected.
- Keep the model and optimizer creation **outside** the training step so variables are reused. Create the tape **inside** the step so each step has a fresh tape.

---

## Common pitfalls

- **Using a Python float instead of a Variable for gradients:** `tape.gradient(y, x)` requires `x` to be a `tf.Variable` (or a trainable model parameter). Constants do not get gradients.
- **Tape used outside scope:** The tape is only valid inside the `with tf.GradientTape() as tape:` block. Do not call `tape.gradient` after the block.
- **Graph vs eager:** In TensorFlow 2, eager execution is default. If you use `tf.function`, ensure inputs are tensors or convert with `tf.convert_to_tensor`; avoid passing Python lists that change shape between calls (they can trigger retracing).

---

**Docs:** [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs). [Keras](https://keras.io/) for high-level API.
