import numpy as np


def wilson_confidence_interval(successes, n_trials, confidence=0.95):
    if n_trials == 0:
        return (0.0, 1.0)
    p_hat = successes / n_trials
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    denom = 1 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_rate_with_ci(successes, n_trials, confidence=0.95):
    rate = successes / n_trials if n_trials > 0 else 0.0
    lower, upper = wilson_confidence_interval(successes, n_trials, confidence)
    return (f"{rate*100:.1f}% (95% CI: {lower*100:.1f}%-{upper*100:.1f}%, n={n_trials})",
            rate, lower, upper)
