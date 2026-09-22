import numpy as np


def generate_pink_noise(n_samples, rng):


  
    white = rng.normal(0, 1, n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0 
    shaped = spectrum / np.sqrt(freqs)
    shaped[0] = 0 
    pink = np.fft.irfft(shaped, n=n_samples)
    pink = pink / (np.std(pink) + 1e-15)  
    return pink


def realistic_sensor_noise(shape, noise_floor_pT, rng,
                            pink_fraction=0.6, common_mode_fraction=0.25):


                              
    n_sensors, n_time = shape

    independent_noise = np.zeros(shape)
    for i in range(n_sensors):
        white_part = rng.normal(0, 1, n_time)
        pink_part = generate_pink_noise(n_time, rng)
        mixed = np.sqrt(1 - pink_fraction) * white_part + np.sqrt(pink_fraction) * pink_part
        independent_noise[i] = mixed


                              
    common_white = rng.normal(0, 1, n_time)
    common_pink = generate_pink_noise(n_time, rng)
    common_mode = np.sqrt(1 - pink_fraction) * common_white + np.sqrt(pink_fraction) * common_pink
    common_mode = np.tile(common_mode, (n_sensors, 1))

    combined = np.sqrt(1 - common_mode_fraction) * independent_noise + \
        np.sqrt(common_mode_fraction) * common_mode


                              
    current_std = combined.std()
    if current_std > 1e-15:
        combined = combined / current_std * noise_floor_pT
    return combined


def jittered_pulse_params(rng, velocity_jitter_frac=0.20, width_jitter_frac=0.20,
                           amplitude_jitter_frac=0.30):

    velocity_mult = 1 + rng.uniform(-velocity_jitter_frac, velocity_jitter_frac)
    width_mult = 1 + rng.uniform(-width_jitter_frac, width_jitter_frac)
    amplitude_mult = 1 + rng.uniform(-amplitude_jitter_frac, amplitude_jitter_frac)
    return velocity_mult, width_mult, amplitude_mult
