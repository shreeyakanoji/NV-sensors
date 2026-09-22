import numpy as np
from axon_field_model import axon_field_pT, noise_floor_pT, I_GIANT_A, I_MAMMALIAN_A

CONDUCTION_VELOCITY_M_S = 2.0
PULSE_WIDTH_M = 1e-3





def traveling_pulse_current(I_peak_A, axon_z_m, t_s, velocity_mult=1.0, width_mult=1.0, amplitude_mult=1.0):
    v = CONDUCTION_VELOCITY_M_S * velocity_mult
    width = PULSE_WIDTH_M * width_mult
    center = v * t_s
    profile = np.exp(-0.5 * ((axon_z_m - center) / (width / 4)) ** 2)
    return I_peak_A * amplitude_mult * profile




def clean_trace(I_peak_A, z_sensor_m, time_points_s, standoff_m,
                velocity_mult=1.0, width_mult=1.0, amplitude_mult=1.0):
    I_t = traveling_pulse_current(I_peak_A, z_sensor_m, time_points_s, velocity_mult, width_mult, amplitude_mult)
    return axon_field_pT(I_t, standoff_m)


def build_array_and_template(I_peak_A, standoff_m, n_sensors=5,
                              array_half_length_m=3e-3, n_time_samples=300,
                              time_half_window_s=2e-3, velocity_mult=1.0,
                              width_mult=1.0, amplitude_mult=1.0):
    sensor_z = np.linspace(-array_half_length_m, array_half_length_m, n_sensors)
    time_points = np.linspace(-time_half_window_s, time_half_window_s, n_time_samples)
    templates = np.array([clean_trace(I_peak_A, z, time_points, standoff_m,
                                        velocity_mult, width_mult, amplitude_mult) for z in sensor_z])
    return sensor_z, time_points, templates



def run_phantom_experiment(I_peak_A, label, standoff_m, n_sensors=5,
                            sensitivity_pT_per_sqrtHz=1.0, bandwidth_hz=1000.0,
                            n_averages=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    sensor_z, time_points, signals = build_array_and_template(I_peak_A, standoff_m, n_sensors)
    noise_floor = noise_floor_pT(sensitivity_pT_per_sqrtHz, bandwidth_hz, n_averages)
    noisy_signals = signals + rng.normal(0, noise_floor, signals.shape)

    peak_signal = signals.max()
    snr = peak_signal / noise_floor if noise_floor > 0 else np.inf
    detected = peak_signal > 3 * noise_floor


                              
    return dict(
        label=label, time_points=time_points, sensor_z=sensor_z,
        clean_signals=signals, noisy_signals=noisy_signals,
        noise_floor_pT=noise_floor, peak_signal_pT=peak_signal,
        snr=snr, detected=detected, standoff_um=standoff_m * 1e6,
    )
