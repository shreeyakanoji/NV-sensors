import numpy as np
from PIL import Image

import axon_field_model as afm
from detection_pipeline import matched_filter_statistic
from statistics_utils import wilson_confidence_interval




def load_image_profile(file_bytes, extraction_mode="middle_row"):
    img = Image.open(file_bytes).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float64)

    if extraction_mode == "middle_row":
        profile = arr[arr.shape[0] // 2, :]
    elif extraction_mode == "middle_column":
        profile = arr[:, arr.shape[1] // 2]
    elif extraction_mode == "diagonal":
        n = min(arr.shape)
        profile = np.array([arr[i, i] for i in range(n)])
    else:
        raise ValueError(f"unknown extraction_mode: {extraction_mode}")

    return profile, arr


def load_nifti_profile(file_path, slice_axis=2, slice_index=None, extraction_mode="middle_row"):
    import nibabel as nib
    img = nib.load(file_path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]

    if slice_index is None:
        slice_index = data.shape[slice_axis] // 2

    if slice_axis == 0:
        slice_2d = data[slice_index, :, :]
    elif slice_axis == 1:
        slice_2d = data[:, slice_index, :]
    else:
        slice_2d = data[:, :, slice_index]

    if extraction_mode == "middle_row":
        profile = slice_2d[slice_2d.shape[0] // 2, :]
    elif extraction_mode == "middle_column":
        profile = slice_2d[:, slice_2d.shape[1] // 2]
    else:
        n = min(slice_2d.shape)
        profile = np.array([slice_2d[i, i] for i in range(n)])

    return profile, slice_2d


def profile_to_current_waveform(raw_profile, peak_current_A):


  
    p = raw_profile - raw_profile.min()
    if p.max() > 0:
        p = p / p.max()
    return p * peak_current_A


def simulate_sensor_response(current_waveform_A, standoff_m, n_resample=300):


  
    x_orig = np.linspace(0, 1, len(current_waveform_A))
    x_new = np.linspace(0, 1, n_resample)
    resampled_current = np.interp(x_new, x_orig, current_waveform_A)
    field_pT = afm.axon_field_pT(np.abs(resampled_current), standoff_m)
    return field_pT


def test_detection_on_uploaded_profile(raw_profile, peak_current_A, standoff_m,
                                        sensitivity_pT_per_sqrtHz=1.0, bandwidth_hz=1000.0,
                                        n_averages=1000, n_trials=100, seed=1):


                                          
    current_waveform = profile_to_current_waveform(raw_profile, peak_current_A)
    true_field_pT = simulate_sensor_response(current_waveform, standoff_m)

    from phantom_experiment import build_array_and_template
    _, _, default_template = build_array_and_template(afm.I_MAMMALIAN_A, standoff_m, n_sensors=1)
    default_template_1d = default_template[0]
    if len(default_template_1d) != len(true_field_pT):
        x_old = np.linspace(0, 1, len(default_template_1d))
        x_new = np.linspace(0, 1, len(true_field_pT))
        default_template_1d = np.interp(x_new, x_old, default_template_1d)

    noise_floor = afm.noise_floor_pT(sensitivity_pT_per_sqrtHz, bandwidth_hz, n_averages)
    rng = np.random.default_rng(seed)

    oracle_h1, oracle_h0, realistic_h1, realistic_h0 = [], [], [], []
    true_field_2d = true_field_pT.reshape(1, -1)
    default_template_2d = default_template_1d.reshape(1, -1)

    for _ in range(n_trials):
        noise1 = rng.normal(0, noise_floor, true_field_2d.shape)
        oracle_stat, _ = matched_filter_statistic(true_field_2d + noise1, true_field_2d)
        oracle_h1.append(oracle_stat)
        noise0 = rng.normal(0, noise_floor, true_field_2d.shape)
        oracle_h0_stat, _ = matched_filter_statistic(noise0, true_field_2d)
        oracle_h0.append(oracle_h0_stat)

        noise2 = rng.normal(0, noise_floor, true_field_2d.shape)
        realistic_stat, _ = matched_filter_statistic(true_field_2d + noise2, default_template_2d)
        realistic_h1.append(realistic_stat)
        noise3 = rng.normal(0, noise_floor, true_field_2d.shape)
        realistic_h0_stat, _ = matched_filter_statistic(noise3, default_template_2d)
        realistic_h0.append(realistic_h0_stat)

    oracle_h1, oracle_h0 = np.array(oracle_h1), np.array(oracle_h0)
    realistic_h1, realistic_h0 = np.array(realistic_h1), np.array(realistic_h0)

    oracle_threshold = np.percentile(oracle_h0, 95)
    realistic_threshold = np.percentile(realistic_h0, 95)

    n_oracle = int(np.sum(oracle_h1 > oracle_threshold))
    n_realistic = int(np.sum(realistic_h1 > realistic_threshold))
    return dict(
        true_field_pT=true_field_pT,
        oracle_detection_rate=n_oracle/n_trials,
        realistic_detection_rate=n_realistic/n_trials,
        oracle_detection_ci_95=wilson_confidence_interval(n_oracle, n_trials),
        realistic_detection_ci_95=wilson_confidence_interval(n_realistic, n_trials),
        n_trials=n_trials,
        oracle_h1=oracle_h1, oracle_h0=oracle_h0,
        realistic_h1=realistic_h1, realistic_h0=realistic_h0,
        peak_field_pT=true_field_pT.max(),
        noise_floor_pT=noise_floor,
    )
