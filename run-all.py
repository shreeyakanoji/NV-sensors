import numpy as np
import matplotlib.pyplot as plt
import json

import axon_field_model as afm
from phantom_experiment import run_phantom_experiment
from detection_pipeline import monte_carlo_detection_test

print("=" * 70)
print("STAGE 1: NV Magnetometry Feasibility for Neural Sensing")
print("=" * 70)


print("\n--- Part 1: Model calibration and validation ---")
print(f"Calibrated giant-axon current (from real Barry et al. 2016 measurement): "
      f"{afm.I_GIANT_A*1e6:.3f} uA")
print(f"Diameter-scaled mammalian current: {afm.I_MAMMALIAN_A*1e9:.3f} nA")
validation = afm.validate_against_independent_literature()
print(f"Cross-check against independent literature (arXiv:2004.14802): "
      f"{validation['predicted_pT']:.2f} pT predicted vs "
      f"{validation['literature_pT']:.2f} pT published "
      f"({validation['agreement_pct']:.1f}% agreement)")



print("\n--- Part 2: Detectability vs standoff distance ---")
distances_um = np.logspace(0, 4.5, 300)
NV_SENSITIVITY = 1.0
SPIKE_BANDWIDTH_HZ = 1000.0

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, n_avg, title in zip(axes1, [1, 1000], ["Single-shot", "1000x averaged"]):
    noise = afm.noise_floor_pT(NV_SENSITIVITY, SPIKE_BANDWIDTH_HZ, n_avg)
    ax.axhline(noise, color="black", linestyle="--", label=f"NV noise floor ({noise:.2f} pT)")
    for I_A, label in [(afm.I_GIANT_A, "Giant axon (calibrated)"),
                         (afm.I_MAMMALIAN_A, "Mammalian axon (scaled)")]:
        B = afm.axon_field_pT(I_A, distances_um * 1e-6)
        ax.plot(distances_um, B, linewidth=2, label=label)
        r_max = afm.max_detectable_distance_um(I_A, noise)
        print(f"  [{title}] {label}: max detectable standoff = {r_max:.1f} um")
    ax.scatter([afm.R_MAMMALIAN_LITERATURE_UM], [afm.B_MAMMALIAN_LITERATURE_PT],
               color="red", zorder=5, s=40, label="Independent lit. check")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Standoff distance (um)"); ax.set_ylabel("Field (pT)")
    ax.set_title(title); ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("outputs_detectability_curves.png", dpi=140)
print("  saved outputs_detectability_curves.png")


print("\n--- Part 3: Simulated phantom experiment @ 500um standoff ---")
rng = np.random.default_rng(42)
phantom_results = {
    "Giant axon @ 500um": run_phantom_experiment(afm.I_GIANT_A, "Giant axon", 500e-6, rng=rng),
    "Mammalian axon @ 500um": run_phantom_experiment(afm.I_MAMMALIAN_A, "Mammalian axon", 500e-6, rng=rng),
}

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
for col, (label, res) in enumerate(phantom_results.items()):
    ax_space = axes2[0, col]
    im = ax_space.imshow(res["noisy_signals"], aspect="auto", cmap="RdBu_r",
                          extent=[res["time_points"][0]*1000, res["time_points"][-1]*1000,
                                  res["sensor_z"][-1]*1000, res["sensor_z"][0]*1000])
    ax_space.set_xlabel("Time (ms)"); ax_space.set_ylabel("Sensor position (mm)")
    ax_space.set_title(f"{label}\nraw sensor array readout")
    plt.colorbar(im, ax=ax_space, label="pT")

    ax_trace = axes2[1, col]
    mid = len(res["sensor_z"]) // 2
    ax_trace.plot(res["time_points"]*1000, res["clean_signals"][mid], label="True signal", linewidth=2)
    ax_trace.plot(res["time_points"]*1000, res["noisy_signals"][mid], alpha=0.5, label="Noisy (measured)")
    ax_trace.axhline(res["noise_floor_pT"], color="black", linestyle="--", label="Noise floor")
    ax_trace.axhline(-res["noise_floor_pT"], color="black", linestyle="--")
    ax_trace.set_title(f"SNR={res['snr']:.2f} | {'DETECTED' if res['detected'] else 'NOT detected (raw)'}")
    ax_trace.set_xlabel("Time (ms)"); ax_trace.set_ylabel("Field (pT)")
    ax_trace.legend(fontsize=8); ax_trace.grid(alpha=0.3)
    print(f"  {label}: SNR={res['snr']:.2f}, raw detection={res['detected']}")
plt.tight_layout()
plt.savefig("outputs_phantom_experiment.png", dpi=140)
print("  saved outputs_phantom_experiment.png")




print("\n--- Part 4: Matched-filter + multi-sensor detection (300 Monte Carlo trials) ---")
detection_results = {
    "Giant axon @ 500um": monte_carlo_detection_test(afm.I_GIANT_A, 500e-6, "Giant axon"),
    "Mammalian @ 500um (hard case)": monte_carlo_detection_test(afm.I_MAMMALIAN_A, 500e-6, "Mammalian"),
    "Mammalian @ 200um (closer)": monte_carlo_detection_test(afm.I_MAMMALIAN_A, 200e-6, "Mammalian closer"),
}

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (label, r) in zip(axes3, detection_results.items()):
    ax.hist(r["h0_stats"], bins=30, alpha=0.6, label="H0 (noise only)", color="gray")
    ax.hist(r["h1_stats"], bins=30, alpha=0.6, label="H1 (signal present)", color="steelblue")
    ax.axvline(r["threshold"], color="red", linestyle="--", label="Threshold (5% FP rate)")
    ax.set_title(f"{label}\ndetection rate: {r['detection_rate']*100:.1f}%")
    ax.set_xlabel("Matched-filter statistic"); ax.legend(fontsize=8)
    print(f"  {label}: raw SNR={r['single_sensor_snr']:.2f} -> "
          f"matched-filter detection rate={r['detection_rate']*100:.1f}% @ "
          f"{r['false_positive_rate']*100:.0f}% false-positive rate")
plt.tight_layout()
plt.savefig("outputs_detection_results.png", dpi=140)
print("  saved outputs_detection_results.png")


summary = {
    "physics_validation": {
        "giant_axon_current_uA": afm.I_GIANT_A * 1e6,
        "mammalian_current_nA": afm.I_MAMMALIAN_A * 1e9,
        "independent_literature_check": validation,
    },
    "detection_results": {
        label: {
            "standoff_um": r["standoff_um"],
            "single_sensor_snr": r["single_sensor_snr"],
            "matched_filter_detection_rate_pct": r["detection_rate"] * 100,
            "false_positive_rate_pct": r["false_positive_rate"] * 100,
        }
        for label, r in detection_results.items()
    },
}
with open("outputs_stage1_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("\nsaved outputs_stage1_summary.json")
print("\n" + "=" * 70)
print("STAGE 1 COMPLETE")
print("=" * 70)
