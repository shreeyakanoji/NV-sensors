import numpy as np
import nibabel as nib

import axon_field_model as afm
from parameter_sweep import detection_result_at
from statistics_utils import wilson_confidence_interval





TISSUE_LABELS = {
    0: "background", 1: "white_matter", 2: "gray_matter", 3: "csf",
    4: "bone", 5: "scalp", 6: "eye_balls", 7: "compact_bone",
    8: "spongy_bone", 9: "blood", 10: "muscle",
}
BRAIN_LABELS = {1, 2} 


def load_tissue_volume(path):
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    voxel_mm = img.header.get_zooms()[:3]
    return data, voxel_mm




def get_slice(data, axis, index):
    if axis == 0:
        return data[index, :, :]
    elif axis == 1:
        return data[:, index, :]
    else:
        return data[:, :, index]



def slice_tissue_centroid(slice_2d):
    ys, xs = np.where(slice_2d > 0)
    return ys.mean(), xs.mean()


def measure_standoff_along_ray(slice_2d, angle_deg, voxel_mm_inplane, max_radius_px=200):
   
    cy, cx = slice_tissue_centroid(slice_2d)
    theta = np.radians(angle_deg)
    direction = np.array([np.sin(theta), np.cos(theta)])  # (dy, dx)

   
    start = np.array([cy, cx]) + direction * max_radius_px
    samples_per_px = 2
    n_samples = max_radius_px * samples_per_px
    path = []
    for i in range(n_samples):
        t = i / samples_per_px
        pos = start - direction * t
        y, x = int(round(pos[0])), int(round(pos[1]))
        if 0 <= y < slice_2d.shape[0] and 0 <= x < slice_2d.shape[1]:
            path.append((y, x, slice_2d[y, x]))

   
    first_tissue_idx = next((i for i, (_, _, v) in enumerate(path) if v > 0), None)
    first_brain_idx = next((i for i, (_, _, v) in enumerate(path) if v in BRAIN_LABELS), None)

    if first_tissue_idx is None or first_brain_idx is None or first_brain_idx <= first_tissue_idx:
        return None 

    step_mm = np.linalg.norm(direction * voxel_mm_inplane) / samples_per_px
    total_standoff_mm = (first_brain_idx - first_tissue_idx) * step_mm

    # tissue breakdown along the path
    breakdown_px = {}
    for i in range(first_tissue_idx, first_brain_idx):
        label = int(path[i][2])
        name = TISSUE_LABELS.get(label, f"label_{label}")
        breakdown_px[name] = breakdown_px.get(name, 0) + 1
    breakdown_mm = {k: round(v * step_mm, 2) for k, v in breakdown_px.items()}

    path_coords = [(p[0], p[1]) for p in path[first_tissue_idx:first_brain_idx + 1]]
    return dict(
        total_standoff_mm=round(total_standoff_mm, 2),
        tissue_breakdown_mm=breakdown_mm,
        path_coords=path_coords,
        scalp_point=path[first_tissue_idx][:2],
        brain_point=path[first_brain_idx][:2],
    )


def recommend_settings_for_standoff(standoff_mm, target_detection_rate=0.90, n_trials=100):
   
    standoff_m = standoff_mm * 1e-3
    sensitivities = {"Levitated magnetometer (0.032 pT/\u221aHz, published)": 0.032,
                      "Best-case NV (0.1 pT/\u221aHz)": 0.1,
                      "Typical NV ensemble (1.0 pT/\u221aHz)": 1.0}
    averaging_options = [1, 10, 100, 1000, 10000]
    signal_cases = {"Mammalian (realistic)": afm.I_MAMMALIAN_A, "Giant axon (proven)": afm.I_GIANT_A}

    results = {}
    for signal_label, I_A in signal_cases.items():
        results[signal_label] = {}
        for sens_label, sens_val in sensitivities.items():
            required_averaging = None
            ci = None
            for n_avg in averaging_options:
                r = detection_result_at(sens_val, standoff_m, n_averages=n_avg, I_A=I_A, n_trials=n_trials)
                if r["detection_rate"] >= target_detection_rate:
                    required_averaging = n_avg
                    ci = r["detection_rate_ci_95"]
                    break
            results[signal_label][sens_label] = dict(
                required_n_averages=required_averaging,
                achievable=required_averaging is not None,
                detection_rate_ci_95=ci,
            )
    return results
