# Volume 1 Starter: Discounted Return
# Chapter 1 exercise

def discounted_return(rewards, gamma=0.9):
    """
    Compute G_0 = sum over t of gamma^t * rewards[t].
    
    Args:
        rewards: list of floats
        gamma: discount factor (float in [0,1])
    
    Returns:
        G: float, the discounted return from step 0
    """
    # TODO: implement using a loop or sum() with enumerate
    pass


def test_discounted_return():
    """Tests for your implementation."""
    assert abs(discounted_return([0, 0, 1], 0.9) - 0.81) < 1e-6, "Test 1 failed"
    assert abs(discounted_return([1, 0, 0], 0.9) - 1.0) < 1e-6, "Test 2 failed"
    assert abs(discounted_return([1, 1, 1], 0.9) - 2.71) < 1e-2, "Test 3 failed"
    assert discounted_return([], 0.9) == 0, "Test 4 failed (empty list)"
    print("All tests passed!")


if __name__ == "__main__":
    test_discounted_return()
    # Example usage
    rewards = [0, 0, 1]
    G = discounted_return(rewards, gamma=0.9)
    print(f"G_0 for {rewards} with gamma=0.9: {G:.4f}")
