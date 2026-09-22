import numpy as np

# ===========================================================================

#
# result for the magnetic field of an axon in an infinite uniform
# volume conductor (Swinney & Wikswo 1980; Roth & Wikswo 1985). For that
# standard, well established idealization, the azimuthal field at radial
# distance r depends ONLY on the intracellular axial current at that same
# point exactly & SIMILARLY like Ampere's law for a simple current-carrying wire:
#
#     B(r) = mu0 * I_intracellular / (2*pi*r)
#
# ===========================================================================

MU0 = 4 * np.pi * 1e-7  # T*m/A, vacuum permeability

# --- Calibration point: a REAL, PUBLISHED measurement ---
# Barry et al. 2016, PNAS ;- NV diamond magnetometry of a squid/worm giant
# axon action potential. This is the only number in this model that comes
# directly from a real experiment; everything else is derived from it.
B_GIANT_MEASURED_PT = 600.0
R_GIANT_MEASURED_UM = 300.0

# --- Physiological diameters (real, published) ---
D_GIANT_AXON_UM = 300.0      # squid/worm giant axon
D_MAMMALIAN_AXON_UM = 1.0    # mammalian axon at nodes of Ranvier

# --- Independent literature check point (NOT used in calibration. 
# used only to validate the model's prediction against a completely
# separate computational study) ---
B_MAMMALIAN_LITERATURE_PT = 36.0   # arXiv:2004.14802, axon hillock model
R_MAMMALIAN_LITERATURE_UM = 20.5


def calibrated_giant_axon_current_A():
   
    r_m = R_GIANT_MEASURED_UM * 1e-6
    B_T = B_GIANT_MEASURED_PT * 1e-12
    return B_T * 2 * np.pi * r_m / MU0


def scaled_mammalian_current_A():
   
    I_giant = calibrated_giant_axon_current_A()
    return I_giant * (D_MAMMALIAN_AXON_UM / D_GIANT_AXON_UM)


I_GIANT_A = calibrated_giant_axon_current_A()
I_MAMMALIAN_A = scaled_mammalian_current_A()


def axon_field_pT(I_A, r_m):
    return (MU0 * I_A / (2 * np.pi * np.maximum(r_m, 1e-9))) * 1e12


def validate_against_independent_literature():
    r_m = R_MAMMALIAN_LITERATURE_UM * 1e-6
    predicted_pT = axon_field_pT(I_MAMMALIAN_A, r_m)
    agreement_pct = 100 * (1 - abs(predicted_pT - B_MAMMALIAN_LITERATURE_PT) / B_MAMMALIAN_LITERATURE_PT)
    return dict(
        predicted_pT=predicted_pT,
        literature_pT=B_MAMMALIAN_LITERATURE_PT,
        distance_um=R_MAMMALIAN_LITERATURE_UM,
        agreement_pct=agreement_pct,
    )


def noise_floor_pT(sensitivity_pT_per_sqrtHz, bandwidth_hz, n_averages=1):
    return (sensitivity_pT_per_sqrtHz * np.sqrt(bandwidth_hz)) / np.sqrt(n_averages)


def max_detectable_distance_um(I_A, noise_floor_pT_val, snr_threshold=1.0):
    target_B_T = (snr_threshold * noise_floor_pT_val) * 1e-12
    if target_B_T <= 0:
        return np.inf
    r_m = MU0 * I_A / (2 * np.pi * target_B_T)
    return r_m * 1e6


if __name__ == "__main__":
    print(f"Calibrated giant-axon current:   {I_GIANT_A*1e6:.3f} uA")
    print(f"Diameter-scaled mammalian current: {I_MAMMALIAN_A*1e9:.3f} nA")
    v = validate_against_independent_literature()
    print(f"\nValidation against independent literature (not used in calibration):")
    print(f"  predicted: {v['predicted_pT']:.2f} pT at {v['distance_um']}um")
    print(f"  published: {v['literature_pT']:.2f} pT")
    print(f"  agreement: {v['agreement_pct']:.1f}%")
