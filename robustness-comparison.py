import numpy as np
import matplotlib.pyplot as plt

import axon_field_model as afm
from phantom_experiment import build_array_and_template
from realistic_noise_model import realistic_sensor_noise, jittered_pulse_params
from statistics_utils import wilson_confidence_interval, format_rate_with_ci




def matched_filter_statistic(noisy_traces, templates):
    per_sensor_stats = []
    for noisy, template in zip(noisy_traces, templates):
        norm = np.linalg.norm(template)
        if norm < 1e-12:
            per_sensor_stats.append(0.0)
            continue
        per_sensor_stats.append(np.dot(noisy, template) / norm)
    return np.sum(per_sensor_stats), np.array(per_sensor_stats)



def monte_carlo_idealized(I_peak_A, standoff_m, n_sensors=5, sensitivity=1.0,
                           bandwidth_hz=1000.0, n_averages=1000, n_trials=300, fpr=0.05, seed=1):
    # the ORIGINAL Stage 1 assumptions: pure white noise, perfect template match
    sensor_z, time_points, templates = build_array_and_template(I_peak_A, standoff_m, n_sensors)
    noise_floor = afm.noise_floor_pT(sensitivity, bandwidth_hz, n_averages)
    rng = np.random.default_rng(seed)

    h1, h0 = [], []
    for _ in range(n_trials):
        h1_stat, _ = matched_filter_statistic(templates + rng.normal(0, noise_floor, templates.shape), templates)
        h0_stat, _ = matched_filter_statistic(rng.normal(0, noise_floor, templates.shape), templates)
        h1.append(h1_stat); h0.append(h0_stat)
    h1, h0 = np.array(h1), np.array(h0)
    threshold = np.percentile(h0, 100 * (1 - fpr))
    n_det = int(np.sum(h1 > threshold))
    return dict(h1=h1, h0=h0, threshold=threshold, detection_rate=n_det/n_trials,
                n_detected=n_det, n_trials=n_trials, detection_rate_ci_95=wilson_confidence_interval(n_det, n_trials))


def monte_carlo_realistic(I_peak_A, standoff_m, n_sensors=5, sensitivity=1.0,
                           bandwidth_hz=1000.0, n_averages=1000, n_trials=300, fpr=0.05, seed=1,
                           pink_fraction=0.6, common_mode_fraction=0.25,
                           enable_jitter=True):
  
    noise_floor = afm.noise_floor_pT(sensitivity, bandwidth_hz, n_averages)
    rng = np.random.default_rng(seed)

   
    _, _, nominal_template = build_array_and_template(I_peak_A, standoff_m, n_sensors)

    h1, h0 = [], []
    for _ in range(n_trials):
        if enable_jitter:
            v_mult, w_mult, a_mult = jittered_pulse_params(rng)
        else:
            v_mult, w_mult, a_mult = 1.0, 1.0, 1.0
        _, _, true_signal = build_array_and_template(
            I_peak_A, standoff_m, n_sensors,
            velocity_mult=v_mult, width_mult=w_mult, amplitude_mult=a_mult)

        noise_h1 = realistic_sensor_noise(true_signal.shape, noise_floor, rng,
                                            pink_fraction, common_mode_fraction)
        h1_stat, _ = matched_filter_statistic(true_signal + noise_h1, nominal_template)
        h1.append(h1_stat)

        noise_h0 = realistic_sensor_noise(true_signal.shape, noise_floor, rng,
                                            pink_fraction, common_mode_fraction)
        h0_stat, _ = matched_filter_statistic(noise_h0, nominal_template)
        h0.append(h0_stat)


                             
    h1, h0 = np.array(h1), np.array(h0)
    threshold = np.percentile(h0, 100 * (1 - fpr))
    n_det = int(np.sum(h1 > threshold))
    return dict(h1=h1, h0=h0, threshold=threshold, detection_rate=n_det/n_trials,
                n_detected=n_det, n_trials=n_trials, detection_rate_ci_95=wilson_confidence_interval(n_det, n_trials))




if __name__ == "__main__":
    print("=== Robustness comparison: idealized (original Stage 1) vs realistic noise/jitter ===\n")

    test_cases = [
        ("Giant axon @ 500um", afm.I_GIANT_A, 500e-6),
        ("Mammalian @ 500um (hard case)", afm.I_MAMMALIAN_A, 500e-6),
        ("Mammalian @ 200um", afm.I_MAMMALIAN_A, 200e-6),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    results_summary = []

    for col, (label, I_A, standoff) in enumerate(test_cases):
        idealized = monte_carlo_idealized(I_A, standoff)
        realistic = monte_carlo_realistic(I_A, standoff)

        results_summary.append((label, idealized["detection_rate"], realistic["detection_rate"]))
        print(f"{label}:")
        print(f"  idealized (white noise, perfect template): {idealized['detection_rate']*100:.1f}% detection")
        print(f"  realistic (1/f + correlated noise + jitter): {realistic['detection_rate']*100:.1f}% detection")
        print()

        ax_top = axes[0, col]
        ax_top.hist(idealized["h0"], bins=30, alpha=0.6, label="H0", color="gray")
        ax_top.hist(idealized["h1"], bins=30, alpha=0.6, label="H1", color="steelblue")
        ax_top.axvline(idealized["threshold"], color="red", linestyle="--")
        ax_top.set_title(f"{label}\nIDEALIZED: {idealized['detection_rate']*100:.1f}% detection")
        ax_top.legend(fontsize=7)

        ax_bot = axes[1, col]
        ax_bot.hist(realistic["h0"], bins=30, alpha=0.6, label="H0", color="gray")
        ax_bot.hist(realistic["h1"], bins=30, alpha=0.6, label="H1", color="darkorange")
        ax_bot.axvline(realistic["threshold"], color="red", linestyle="--")
        ax_bot.set_title(f"REALISTIC: {realistic['detection_rate']*100:.1f}% detection")
        ax_bot.set_xlabel("Matched-filter statistic")
        ax_bot.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("outputs_robustness_comparison.png", dpi=140)
    print("saved outputs_robustness_comparison.png")

    print("\n=== Summary ===")
    for label, ideal_rate, real_rate in results_summary:
        drop = ideal_rate - real_rate
        print(f"{label}: {ideal_rate*100:.1f}% -> {real_rate*100:.1f}% "
              f"(drop of {drop*100:.1f} percentage points)")
