import numpy as np
from phantom_experiment import build_array_and_template
from axon_field_model import noise_floor_pT
from statistics_utils import wilson_confidence_interval


def matched_filter_statistic(noisy_traces, templates):
    per_sensor_stats = []
    for noisy, template in zip(noisy_traces, templates):
        norm = np.linalg.norm(template)
        if norm < 1e-12:
            per_sensor_stats.append(0.0)
            continue
        stat = np.dot(noisy, template) / norm
        per_sensor_stats.append(stat)
    combined = np.sum(per_sensor_stats)
    return combined, np.array(per_sensor_stats)




def monte_carlo_detection_test(I_peak_A, standoff_m, label, n_sensors=5,
                                sensitivity_pT_per_sqrtHz=1.0, bandwidth_hz=1000.0,
                                n_averages=1000, n_trials=300, false_positive_rate=0.05,
                                seed=123):
    sensor_z, time_points, templates = build_array_and_template(I_peak_A, standoff_m, n_sensors)
    noise_floor = noise_floor_pT(sensitivity_pT_per_sqrtHz, bandwidth_hz, n_averages)
    rng = np.random.default_rng(seed)

    single_sensor_snr = templates[len(templates) // 2].max() / noise_floor

    h1_stats, h0_stats = [], []
    for _ in range(n_trials):
        noise = rng.normal(0, noise_floor, templates.shape)
        h1_combined, _ = matched_filter_statistic(templates + noise, templates)
        h1_stats.append(h1_combined)

        noise_only = rng.normal(0, noise_floor, templates.shape)
        h0_combined, _ = matched_filter_statistic(noise_only, templates)
        h0_stats.append(h0_combined)

    h1_stats, h0_stats = np.array(h1_stats), np.array(h0_stats)
    threshold = np.percentile(h0_stats, 100 * (1 - false_positive_rate))
    n_detected = int(np.sum(h1_stats > threshold))
    detection_rate = n_detected / n_trials
    ci = wilson_confidence_interval(n_detected, n_trials)

                                  
    return dict(
        label=label, standoff_um=standoff_m * 1e6, single_sensor_snr=single_sensor_snr,
        h1_stats=h1_stats, h0_stats=h0_stats, threshold=threshold,
        detection_rate=detection_rate, false_positive_rate=false_positive_rate,
        mean_h1=h1_stats.mean(), mean_h0=h0_stats.mean(),
        n_detected=n_detected, n_trials=n_trials, detection_rate_ci_95=ci,
    )
