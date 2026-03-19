# Volume 5 Starter: PPO Clipped Objective
# Chapter 43 exercise — Core PPO clipping logic (no neural network)
import numpy as np
import matplotlib.pyplot as plt


def ppo_clipped_objective(ratio, advantage, epsilon=0.2):
    """
    PPO clipped surrogate objective for a single transition.

    L_CLIP = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Args:
        ratio: float, π_θ(a|s) / π_θ_old(a|s)
        advantage: float, A(s,a) estimate
        epsilon: float, clipping range (default 0.2)

    Returns:
        objective: float (we want to MAXIMIZE this)
    """
    # TODO: implement the clipped objective
    clipped = None   # clip(ratio, 1-epsilon, 1+epsilon)
    obj = None       # min(ratio * advantage, clipped * advantage)
    return obj


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).

    GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards: list of rewards
        values: list of value estimates V(s_t) (length T+1, last is V(s_T+1)=0 if done)
        dones: list of done flags
        gamma, lam: discount and GAE lambda

    Returns:
        advantages: numpy array of GAE estimates
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        # TODO: delta = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        # gae = delta + gamma * lam * (1 - done) * gae
        # advantages[t] = gae
        pass

    return advantages


def test_ppo_clipping():
    """Verify PPO clipping behavior."""
    eps = 0.2

    # Case 1: A > 0, ratio too high (1.4) -> should clip to 1.2
    obj = ppo_clipped_objective(1.4, 1.0, eps)
    assert abs(obj - 1.2) < 1e-6, f"Expected 1.2, got {obj}"

    # Case 2: A > 0, ratio OK (1.1) -> no clipping
    obj = ppo_clipped_objective(1.1, 1.0, eps)
    assert abs(obj - 1.1) < 1e-6, f"Expected 1.1, got {obj}"

    # Case 3: A < 0, ratio too low (0.7) -> should clip to 0.8
    obj = ppo_clipped_objective(0.7, -1.0, eps)
    assert abs(obj - 0.8 * (-1.0) - 0) < 1e-6 or abs(obj - (-0.8)) < 1e-6, f"Expected -0.8, got {obj}"

    print("All PPO clipping tests passed!")


def test_gae():
    """Test GAE on a simple example."""
    rewards = [0, 0, 1]
    values = [0.3, 0.4, 0.5, 0.0]   # last value is 0 (done)
    dones = [False, False, True]

    advantages = compute_gae(rewards, values, dones, gamma=0.99, lam=1.0)
    # With lambda=1, GAE = MC return - V(s_t)
    # G_0 = 0 + 0 + 1 = 1. A_0 = G_0 - V_0 = 1 - 0.3 = 0.7
    print(f"GAE advantages: {np.round(advantages, 4)}")
    print(f"Expected A_0 ≈ {1*0.99**0 + 0*0.99 + 1*0.99**2 - 0.3:.4f} (with lambda=1)")


if __name__ == "__main__":
    test_ppo_clipping()
    test_gae()

    # Visualize PPO objective as a function of ratio
    ratios = np.linspace(0.5, 1.8, 100)
    eps = 0.2
    A_pos = 1.0   # positive advantage
    A_neg = -1.0  # negative advantage

    obj_pos = [ppo_clipped_objective(r, A_pos, eps) for r in ratios]
    obj_neg = [ppo_clipped_objective(r, A_neg, eps) for r in ratios]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ratios, obj_pos, label='PPO (clipped)', linewidth=2)
    plt.plot(ratios, ratios * A_pos, '--', label='Unclipped', alpha=0.5)
    plt.axvline(1 - eps, color='red', linestyle=':', label=f'clip bounds')
    plt.axvline(1 + eps, color='red', linestyle=':')
    plt.xlabel('Ratio r(θ)')
    plt.ylabel('Objective')
    plt.title('PPO Clipped Objective (A > 0)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ratios, obj_neg, label='PPO (clipped)', linewidth=2, color='orange')
    plt.plot(ratios, ratios * A_neg, '--', label='Unclipped', alpha=0.5, color='blue')
    plt.axvline(1 - eps, color='red', linestyle=':', label=f'clip bounds')
    plt.axvline(1 + eps, color='red', linestyle=':')
    plt.xlabel('Ratio r(θ)')
    plt.ylabel('Objective')
    plt.title('PPO Clipped Objective (A < 0)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ppo_clipping.png')
    plt.show()
    print("Plot saved to ppo_clipping.png")
