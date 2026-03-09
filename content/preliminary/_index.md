---
title: "Preliminary Assessment"
description: "25 questions to check readiness for the RL curriculum. Solutions included."
date: 2026-03-10T00:00:00Z
draft: false
---

**Who is this for?** This assessment checks whether you have the math, Python, NumPy, PyTorch, and basic RL concepts needed to start the curriculum comfortably.

**New to programming?** Start with the [Learning path](/learning-path/) Phase 0. **Unsure about math?** Try the [Math for RL](/math-for-rl/) track first. After that, use this assessment to see if you are ready for the curriculum.

25 questions to assess your foundational knowledge before the 100-chapter reinforcement learning curriculum. Answer honestly; solutions are below to help you identify gaps. If you can answer at least 20 correctly and feel comfortable with the concepts, you are ready to start. If you struggled with many, review [Prerequisites](/prerequisites/) or the [Learning path](/learning-path/) and come back.

---

### 1. Probability & Statistics

**Q:** In a multi-armed bandit with 3 arms, the true reward distributions are \\(\mathcal{N}(0,1), \mathcal{N}(1,1), \mathcal{N}(-0.5,1)\\). If you pull arm 2 five times, you observe rewards [1.2, 0.8, 1.5, 0.3, 2.1]. What is the sample mean estimate of arm 2's expected reward? What is the unbiased sample variance?

{{< collapse summary="Answer" >}}
Sample mean = (1.2+0.8+1.5+0.3+2.1)/5 = 5.9/5 = **1.18**.

Sample variance = \\(\frac{1}{n-1}\sum (x_i - \bar{x})^2\\) = deviations squared: (1.2-1.18)² + (0.8-1.18)² + (1.5-1.18)² + (0.3-1.18)² + (2.1-1.18)² = 0.0004 + 0.1444 + 0.1024 + 0.7744 + 0.8464 = 1.868; variance = 1.868/4 = **0.467**.
{{< /collapse >}}

---

### 2. Probability & Statistics

**Q:** What is the difference between the expected value of a random variable and a sample average? When do they coincide (in the limit)?

{{< collapse summary="Answer" >}}
The expected value is a theoretical long-run average based on the distribution; the sample average is an empirical estimate from observed data. By the law of large numbers, the sample average converges to the expected value as the number of samples goes to infinity.
{{< /collapse >}}

---

### 3. Linear Algebra

**Q:** Given two vectors \\(x = [1,2,3]^T\\) and \\(y = [4,5,6]^T\\), compute their dot product. What is the geometric interpretation?

{{< collapse summary="Answer" >}}
Dot product = 1·4 + 2·5 + 3·6 = 4+10+18 = **32**. Geometrically, it measures the cosine of the angle between them times the product of their magnitudes.
{{< /collapse >}}

---

### 4. Linear Algebra

**Q:** If \\(A\\) is a matrix and \\(w\\) is a weight vector, what is \\(\nabla_w (A w)\\)? Assume \\(A\\) is constant.

{{< collapse summary="Answer" >}}
\\(\nabla_w (A w) = A^T\\) (gradient of a vector-valued function; in numerator layout it's \\(A\\) itself). For a scalar loss \\(L = f(Aw)\\), the gradient involves \\(A^T\\) times the derivative of \\(f\\).
{{< /collapse >}}

---

### 5. Calculus

**Q:** Compute the derivative of \\(f(x) = \ln(1+e^x)\\) with respect to \\(x\\). This function appears in logistic regression and softmax.

{{< collapse summary="Answer" >}}
\\(f'(x) = \frac{e^x}{1+e^x} = \sigma(x)\\) (the sigmoid function).
{{< /collapse >}}

---

### 6. Calculus

**Q:** What is the chain rule for derivatives? Give an example: if \\(y = f(u)\\) and \\(u = g(x)\\), express \\(\frac{dy}{dx}\\).

{{< collapse summary="Answer" >}}
\\(\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}\\). Example: \\(y = \sin(x^2)\\) → \\(\frac{dy}{dx} = \cos(x^2) \cdot 2x\\).
{{< /collapse >}}

---

### 7. Python Basics

**Q:** Write a Python function that takes a list of numbers and returns the moving average with window size 3. For example, input [1,2,3,4,5] returns [2.0, 3.0, 4.0].

{{< collapse summary="Answer" >}}
```python
def moving_average(arr, window=3):
    return [sum(arr[i:i+window])/window for i in range(len(arr)-window+1)]
```
{{< /collapse >}}

---

### 8. NumPy

**Q:** Create a 3×3 NumPy array of zeros, then set the first row to [1,2,3]. How do you compute the element-wise product of this array with itself?

{{< collapse summary="Answer" >}}
```python
import numpy as np
arr = np.zeros((3,3))
arr[0] = [1,2,3]
prod = arr * arr   # or np.square(arr)
```
{{< /collapse >}}

---

### 9. PyTorch Basics

**Q:** In PyTorch, how do you create a tensor requiring gradient, and how do you compute the gradient of \\(y = x^2\\) with respect to \\(x\\) for \\(x=2.0\\)?

{{< collapse summary="Answer" >}}
```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # should be 4.0
```
{{< /collapse >}}

---

### 10. RL Framework

**Q:** Define the four main components of a reinforcement learning system: agent, environment, action, reward. Also, what is a state?

{{< collapse summary="Answer" >}}
- **Agent:** the learner/decision maker.  
- **Environment:** everything outside the agent that it interacts with.  
- **Action:** a move the agent can make.  
- **Reward:** a scalar feedback signal from the environment indicating the immediate desirability of the agent's state/action.  
- **State:** a representation of the current situation used by the agent to decide actions.
{{< /collapse >}}

---

### 11. Markov Property

**Q:** What is the Markov property in the context of RL? Why is it important?

{{< collapse summary="Answer" >}}
The Markov property states that the future is independent of the past given the present state. In RL, it means the state contains all relevant information for decision making, allowing the agent to ignore history. It's important because it simplifies the problem to a Markov Decision Process.
{{< /collapse >}}

---

### 12. Exploration vs. Exploitation

**Q:** Give a real-world example of the exploration-exploitation dilemma and explain why it's challenging.

{{< collapse summary="Answer" >}}
Example: Choosing a restaurant. Exploitation means going to a known favorite; exploration means trying a new one to potentially find a better option. The challenge is balancing short-term satisfaction with long-term discovery.
{{< /collapse >}}

---

### 13. Discount Factor

**Q:** What is the purpose of a discount factor \\(\gamma\\) in RL? What happens when \\(\gamma=0\\) and when \\(\gamma=1\\) (in continuing tasks)?

{{< collapse summary="Answer" >}}
The discount factor determines the present value of future rewards: it makes the sum finite in continuing tasks and models uncertainty/time preference. \\(\gamma=0\\) makes the agent myopic (only immediate reward matters). \\(\gamma=1\\) treats future rewards equally, which can cause infinite sums in continuing tasks unless episodes terminate.
{{< /collapse >}}

---

### 14. Value Functions

**Q:** Define the state-value function \\(V^\pi(s)\\) and the action-value function \\(Q^\pi(s,a)\\).

{{< collapse summary="Answer" >}}
\\(V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]\\) — the expected return starting from state \\(s\\) and following policy \\(\pi\\).  

\\(Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]\\) — expected return starting from state \\(s\\), taking action \\(a\\), then following \\(\pi\\).
{{< /collapse >}}

---

### 15. Bellman Equation

**Q:** Write the Bellman expectation equation for \\(V^\pi(s)\\) in terms of rewards and next-state values.

{{< collapse summary="Answer" >}}
\\(V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma V^\pi(s')]\\).
{{< /collapse >}}

---

### 16. Dynamic Programming

**Q:** Name two dynamic programming methods for solving MDPs when the model (transition probabilities) is known.

{{< collapse summary="Answer" >}}
**Policy Iteration** and **Value Iteration**.
{{< /collapse >}}

---

### 17. Monte Carlo vs. TD

**Q:** What is the key difference between Monte Carlo and Temporal Difference (TD) learning in terms of updating value estimates?

{{< collapse summary="Answer" >}}
Monte Carlo waits until the end of an episode to compute the return and updates using that full return. TD uses bootstrapping: it updates using the current estimate of the next state's value and the immediate reward, without waiting for the episode to finish.
{{< /collapse >}}

---

### 18. On-policy vs. Off-policy

**Q:** Explain the difference between on-policy and off-policy learning. Give one algorithm example for each.

{{< collapse summary="Answer" >}}
**On-policy** learns about the policy being executed (e.g. SARSA). **Off-policy** learns about a target policy while following a different behavior policy (e.g. Q-learning).
{{< /collapse >}}

---

### 19. Q-learning Update

**Q:** Write the Q-learning update rule for a transition \\((s, a, r, s')\\).

{{< collapse summary="Answer" >}}
\\(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]\\).
{{< /collapse >}}

---

### 20. Function Approximation

**Q:** Why is function approximation necessary in RL for large or continuous state spaces?

{{< collapse summary="Answer" >}}
Tabular methods store a value for each state or state-action pair, which becomes infeasible when the number of states is huge or infinite. Function approximation generalizes from seen states to unseen ones using a parameterized function (e.g. neural network).
{{< /collapse >}}

---

### 21. Gradient Descent

**Q:** In supervised learning, you minimize a loss function \\(L(\theta)\\) using gradient descent: \\(\theta \leftarrow \theta - \alpha \nabla_\theta L\\). What is the analogous update in policy gradient methods?

{{< collapse summary="Answer" >}}
In policy gradient we **maximize** expected return: \\(\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)\\), where \\(J\\) is the expected return and the gradient is given by the policy gradient theorem.
{{< /collapse >}}

---

### 22. Exploration in Deep RL

**Q:** Name two common exploration strategies used in Deep Q-Networks.

{{< collapse summary="Answer" >}}
**ε-greedy** (with probability ε take a random action) and **Noisy Networks** (adding learnable noise to network weights).
{{< /collapse >}}

---

### 23. Experience Replay

**Q:** Why is experience replay used in DQN? What problem does it solve?

{{< collapse summary="Answer" >}}
Experience replay stores transitions in a buffer and samples randomly to break the correlation between consecutive updates, reducing variance and stabilizing training. It also improves sample efficiency by reusing past experiences.
{{< /collapse >}}

---

### 24. Actor-Critic

**Q:** What is the advantage of using an actor-critic method over pure policy gradient (REINFORCE)?

{{< collapse summary="Answer" >}}
Actor-critic methods use a value function (critic) to reduce the variance of policy gradient estimates, often leading to faster and more stable learning. The critic provides a baseline or an advantage estimate.
{{< /collapse >}}

---

### 25. Final Self-Assessment

**Q:** On a scale of 1–10, how comfortable are you with: Python programming (including NumPy and PyTorch)? Probability (expectations, variances, distributions)? Linear algebra (vectors, matrices, gradients)? Calculus (derivatives, chain rule, partial derivatives)? If any area is below 7, consider reviewing before diving into the curriculum.

{{< collapse summary="Answer" >}}
Self-reflection only. The first volume will solidify these foundations, but prior comfort helps. Use [Prerequisites](/prerequisites/) to strengthen weak areas.
{{< /collapse >}}
