import numpy as np
import matplotlib.pyplot as plt

import axon_field_model as afm
from robustness_comparison import monte_carlo_realistic



N_TRIALS = 150  


def detection_result_at(sensitivity_pT_per_sqrtHz, standoff_m, n_averages=1000,
                         n_sensors=5, I_A=None, seed=1, n_trials=N_TRIALS):
    if I_A is None:
        I_A = afm.I_MAMMALIAN_A
    return monte_carlo_realistic(
        I_A, standoff_m, n_sensors=n_sensors,
        sensitivity=sensitivity_pT_per_sqrtHz, n_averages=n_averages,
        n_trials=n_trials, seed=seed,
    )


def detection_rate_at(sensitivity_pT_per_sqrtHz, standoff_m, n_averages=1000,
                       n_sensors=5, I_A=None, seed=1):
    return detection_result_at(sensitivity_pT_per_sqrtHz, standoff_m, n_averages,
                                n_sensors, I_A, seed)["detection_rate"]


def sweep_sensitivity_vs_standoff(sensitivities, standoffs_um, n_averages=1000, n_sensors=5):
    grid = np.zeros((len(sensitivities), len(standoffs_um)))
    for i, sens in enumerate(sensitivities):
        for j, standoff_um in enumerate(standoffs_um):
            grid[i, j] = detection_rate_at(sens, standoff_um * 1e-6, n_averages, n_sensors)
    return grid


def sweep_averaging_vs_standoff(n_averages_list, standoffs_um, sensitivity=1.0, n_sensors=5):
    grid = np.zeros((len(n_averages_list), len(standoffs_um)))
    for i, n_avg in enumerate(n_averages_list):
        for j, standoff_um in enumerate(standoffs_um):
            grid[i, j] = detection_rate_at(sensitivity, standoff_um * 1e-6, n_avg, n_sensors)
    return grid


def sweep_n_sensors_vs_standoff(n_sensors_list, standoffs_um, sensitivity=1.0, n_averages=1000):
    grid = np.zeros((len(n_sensors_list), len(standoffs_um)))
    for i, n_sens in enumerate(n_sensors_list):
        for j, standoff_um in enumerate(standoffs_um):
            grid[i, j] = detection_rate_at(sensitivity, standoff_um * 1e-6, n_averages, n_sens)
    return grid


if __name__ == "__main__":
    standoffs_um = np.array([100, 200, 300, 500, 700, 1000, 1500, 2000])

    print("=== Sweep 1: Sensitivity vs standoff (fixed 1000x averaging, 5 sensors) ===")
    sensitivities = np.array([0.032, 0.1, 0.3, 1.0, 3.0])  # includes the levitated-magnetometer number (0.032)
    grid1 = sweep_sensitivity_vs_standoff(sensitivities, standoffs_um)
    print("done")

    print("\n=== Sweep 2: Averaging vs standoff (fixed NV sensitivity=1.0 pT/sqrt(Hz), 5 sensors) ===")
    n_averages_list = np.array([1, 10, 100, 1000, 10000])
    grid2 = sweep_averaging_vs_standoff(n_averages_list, standoffs_um)
    print("done")

    print("\n=== Sweep 3: Sensor count vs standoff (fixed NV sensitivity, 1000x averaging) ===")
    n_sensors_list = np.array([1, 3, 5, 8, 12])
    grid3 = sweep_n_sensors_vs_standoff(n_sensors_list, standoffs_um)
    print("done")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    im1 = axes[0].imshow(grid1, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                          extent=[0, len(standoffs_um), len(sensitivities), 0])
    axes[0].set_xticks(np.arange(len(standoffs_um)) + 0.5)
    axes[0].set_xticklabels(standoffs_um)
    axes[0].set_yticks(np.arange(len(sensitivities)) + 0.5)
    axes[0].set_yticklabels([f"{s:.3f}" for s in sensitivities])
    axes[0].set_xlabel("Standoff (um)"); axes[0].set_ylabel("Sensitivity (pT/sqrt(Hz))")
    axes[0].set_title("Sensitivity vs standoff\n(1000x avg, 5 sensors)")
    plt.colorbar(im1, ax=axes[0], label="Detection rate")

    im2 = axes[1].imshow(grid2, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                          extent=[0, len(standoffs_um), len(n_averages_list), 0])
    axes[1].set_xticks(np.arange(len(standoffs_um)) + 0.5)
    axes[1].set_xticklabels(standoffs_um)
    axes[1].set_yticks(np.arange(len(n_averages_list)) + 0.5)
    axes[1].set_yticklabels(n_averages_list)
    axes[1].set_xlabel("Standoff (um)"); axes[1].set_ylabel("N averages")
    axes[1].set_title("Averaging vs standoff\n(NV sensitivity, 5 sensors)")
    plt.colorbar(im2, ax=axes[1], label="Detection rate")

    im3 = axes[2].imshow(grid3, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                          extent=[0, len(standoffs_um), len(n_sensors_list), 0])
    axes[2].set_xticks(np.arange(len(standoffs_um)) + 0.5)
    axes[2].set_xticklabels(standoffs_um)
    axes[2].set_yticks(np.arange(len(n_sensors_list)) + 0.5)
    axes[2].set_yticklabels(n_sensors_list)
    axes[2].set_xlabel("Standoff (um)"); axes[2].set_ylabel("N sensors")
    axes[2].set_title("Sensor count vs standoff\n(NV sensitivity, 1000x avg)")
    plt.colorbar(im3, ax=axes[2], label="Detection rate")

    plt.tight_layout()
    plt.savefig("outputs_parameter_sweeps.png", dpi=140)
    print("\nsaved outputs_parameter_sweeps.png")

    import json
    with open("outputs_parameter_sweep_data.json", "w") as f:
        json.dump({
            "sweep1_sensitivity_vs_standoff": {
                "sensitivities_pT_sqrtHz": sensitivities.tolist(),
                "standoffs_um": standoffs_um.tolist(),
                "detection_rate_grid": grid1.tolist(),
            },
            "sweep2_averaging_vs_standoff": {
                "n_averages": n_averages_list.tolist(),
                "standoffs_um": standoffs_um.tolist(),
                "detection_rate_grid": grid2.tolist(),
            },
            "sweep3_sensors_vs_standoff": {
                "n_sensors": n_sensors_list.tolist(),
                "standoffs_um": standoffs_um.tolist(),
                "detection_rate_grid": grid3.tolist(),
            },
        }, f, indent=2)
    print("saved outputs_parameter_sweep_data.json")
