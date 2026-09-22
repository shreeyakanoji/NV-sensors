import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



import axon_field_model as afm
from phantom_experiment import build_array_and_template
from robustness_comparison import monte_carlo_realistic, monte_carlo_idealized
import scan_upload_module as scan_mod
import patient_anatomy_analysis as paa

st.set_page_config(page_title="Quantum Neural Sensor Explorer", layout="wide")
st.title("Quantum Neural Sensor — Settings Explorer")
st.caption("Interactive feasibility model: axon magnetic field physics, calibrated to real "
           "published measurements, with realistic noise and detection modeling.")

tab_sim, tab_upload, tab_anatomy = st.tabs(["Simulated signals", "Upload a real scan", "Personalize to real anatomy"])



with tab_sim:
    with st.sidebar:
        st.header("Signal source")
        case = st.radio("Axon case", ["Mammalian (realistic, hard case)", "Giant axon (squid/worm, proven)"])
        I_A = afm.I_MAMMALIAN_A if "Mammalian" in case else afm.I_GIANT_A

        st.header("Sensor settings")
        sensitivity = st.select_slider(
            "Sensitivity (pT/\u221aHz)",
            options=[0.032, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0],
            value=1.0,
            help="0.032 = published levitated-magnetometer figure (Aug 2026). "
                 "1.0 = typical bulk NV ensemble.",
        )
        standoff_um = st.slider("Standoff distance (\u00b5m)", 50, 3000, 500, step=50)
        n_averages = st.select_slider("Number of averaged repeats", options=[1, 10, 100, 1000, 10000], value=1000)
        n_sensors = st.slider("Number of sensors in array", 1, 15, 5)

        st.header("Realistic noise settings")
        enable_realistic = st.checkbox("Use realistic noise (1/f + correlated + jitter)", value=True)
        pink_fraction = st.slider("1/f (pink) noise fraction", 0.0, 1.0, 0.6, step=0.05, disabled=not enable_realistic)
        common_mode_fraction = st.slider("Correlated cross-sensor noise fraction", 0.0, 1.0, 0.25, step=0.05,
                                           disabled=not enable_realistic)
        enable_jitter = st.checkbox("Enable biological trial-to-trial jitter", value=True, disabled=not enable_realistic)

        n_trials = st.select_slider("Monte Carlo trials (more = slower, more precise)",
                                      options=[50, 100, 200, 300], value=100)
        run_button = st.button("Run simulation", type="primary")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Detectability vs standoff")
        distances_um = np.logspace(0, 4.5, 200)
        noise = afm.noise_floor_pT(sensitivity, 1000.0, n_averages)
        B = afm.axon_field_pT(I_A, distances_um * 1e-6)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(distances_um, B, linewidth=2, color="#1C7293", label="Signal")
        ax.axhline(noise, color="black", linestyle="--", label=f"Noise floor ({noise:.3f} pT)")
        ax.axvline(standoff_um, color="red", linestyle=":", label=f"Current standoff ({standoff_um}\u00b5m)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Standoff (\u00b5m)"); ax.set_ylabel("Field (pT)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        st.pyplot(fig)

        r_max = afm.max_detectable_distance_um(I_A, noise)
        st.metric("Max detectable standoff (SNR=1 threshold)", f"{r_max:.0f} \u00b5m ({r_max/1000:.2f} mm)")

    with col2:
        st.subheader("Monte Carlo detection test")
        if run_button:
            with st.spinner("Running Monte Carlo trials..."):
                if enable_realistic:
                    result = monte_carlo_realistic(
                        I_A, standoff_um * 1e-6, n_sensors=n_sensors,
                        sensitivity=sensitivity, n_averages=n_averages, n_trials=n_trials,
                        pink_fraction=pink_fraction, common_mode_fraction=common_mode_fraction,
                        enable_jitter=enable_jitter,
                    )
                else:
                    result = monte_carlo_idealized(
                        I_A, standoff_um * 1e-6, n_sensors=n_sensors,
                        sensitivity=sensitivity, n_averages=n_averages, n_trials=n_trials,
                    )

            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            ax2.hist(result["h0"], bins=25, alpha=0.6, label="H0 (noise only)", color="gray")
            ax2.hist(result["h1"], bins=25, alpha=0.6, label="H1 (signal present)", color="#065A82")
            ax2.axvline(result["threshold"], color="red", linestyle="--", label="Detection threshold")
            ax2.set_xlabel("Matched-filter statistic"); ax2.legend(fontsize=8)
            st.pyplot(fig2)

            ci_lo, ci_hi = result["detection_rate_ci_95"]
            st.metric("Detection rate (5% false-positive rate)", f"{result['detection_rate']*100:.1f}%",
                      help=f"95% CI: {ci_lo*100:.1f}%-{ci_hi*100:.1f}% (n={result['n_trials']})")
        else:
            st.info("Adjust settings on the left, then click **Run simulation**.")

    st.divider()
    st.caption(
        "Calibrated to a real 2016 NV-diamond measurement of a squid/worm giant axon action potential. "
        "Mammalian-scale current is derived by diameter-based physiological scaling, independently "
        "cross-validated (81.3% agreement) against a separate published computational estimate. "
        "All detection results use matched filtering + coherent multi-sensor combination."
    )

# ===========================================================================
# TAB 2: upload a real scan and test detection against its extracted shape
# ===========================================================================
with tab_upload:
    st.warning(
        "**Honesty note:** this does NOT perform real nerve/axon segmentation on your upload. "
        "It extracts a 1D pixel-intensity profile along a chosen line, treats that profile's "
        "*shape* as a stand-in current waveform (scaled to a peak current you specify), and tests "
        "how the detection pipeline handles a real, arbitrary signal shape instead of the clean "
        "assumed Gaussian pulse used elsewhere in this project. It is not a claim that the profile "
        "corresponds to a real electrophysiological signal.",
    )

    uploaded_file = st.file_uploader("Upload an image (PNG/JPG) or NIfTI file (.nii, .nii.gz)",
                                       type=["png", "jpg", "jpeg", "nii", "gz"])

    if uploaded_file is not None:
        is_nifti = uploaded_file.name.endswith((".nii", ".nii.gz"))

        col_a, col_b = st.columns([1, 1])
        with col_a:
            extraction_mode = st.selectbox("Profile extraction line", ["middle_row", "middle_column", "diagonal"])
            peak_current_nA = st.slider("Assumed peak current (nA)", 0.1, 1000.0, 3.0, step=0.1,
                                          help="3nA \u2248 our mammalian-axon estimate. 900nA \u2248 our giant-axon estimate.")
            standoff_um_upload = st.slider("Sensor standoff (\u00b5m)", 50, 3000, 500, step=50, key="upload_standoff")
            sensitivity_upload = st.select_slider("Sensitivity (pT/\u221aHz)",
                                                    options=[0.032, 0.1, 0.3, 1.0, 3.0], value=1.0, key="upload_sens")
            n_averages_upload = st.select_slider("N averages", options=[1, 10, 100, 1000, 10000],
                                                   value=1000, key="upload_avg")
            analyze_button = st.button("Analyze uploaded scan", type="primary")

        with col_b:
            if is_nifti:
                st.info("NIfTI file detected. Saving temporarily to read slice data...")
                import tempfile, os
                suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    profile, slice_2d = scan_mod.load_nifti_profile(tmp_path, extraction_mode=extraction_mode)
                    fig_slice, ax_slice = plt.subplots(figsize=(5, 5))
                    ax_slice.imshow(slice_2d.T, cmap="gray", origin="lower")
                    ax_slice.set_title("Middle slice (extraction line shown)")
                    if extraction_mode == "middle_row":
                        ax_slice.axhline(slice_2d.shape[0] // 2, color="red", linewidth=1)
                    elif extraction_mode == "middle_column":
                        ax_slice.axvline(slice_2d.shape[1] // 2, color="red", linewidth=1)
                    st.pyplot(fig_slice)
                finally:
                    os.unlink(tmp_path)
            else:
                profile, arr = scan_mod.load_image_profile(uploaded_file, extraction_mode=extraction_mode)
                fig_img, ax_img = plt.subplots(figsize=(5, 5))
                ax_img.imshow(arr, cmap="gray")
                ax_img.set_title("Uploaded image (extraction line shown)")
                if extraction_mode == "middle_row":
                    ax_img.axhline(arr.shape[0] // 2, color="red", linewidth=1)
                elif extraction_mode == "middle_column":
                    ax_img.axvline(arr.shape[1] // 2, color="red", linewidth=1)
                st.pyplot(fig_img)

        if analyze_button:
            with st.spinner("Running detection analysis on extracted profile..."):
                result = scan_mod.test_detection_on_uploaded_profile(
                    profile, peak_current_A=peak_current_nA * 1e-9,
                    standoff_m=standoff_um_upload * 1e-6,
                    sensitivity_pT_per_sqrtHz=sensitivity_upload,
                    n_averages=n_averages_upload, n_trials=100,
                )

            st.subheader("Extracted signal and detection results")
            fig3, (ax_prof, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax_prof.plot(result["true_field_pT"], color="#065A82")
            ax_prof.set_title(f"Simulated field from extracted profile\npeak={result['peak_field_pT']:.3f} pT, "
                               f"noise floor={result['noise_floor_pT']:.3f} pT")
            ax_prof.set_xlabel("Sample index"); ax_prof.set_ylabel("Field (pT)")

            ax_hist.hist(result["realistic_h0"], bins=25, alpha=0.6, label="H0 (noise only)", color="gray")
            ax_hist.hist(result["realistic_h1"], bins=25, alpha=0.6, label="H1 (real signal, default template)", color="orange")
            ax_hist.set_title("Detection using our DEFAULT (mismatched) template")
            ax_hist.legend(fontsize=8)
            st.pyplot(fig3)

            m1, m2 = st.columns(2)
            oci = result["oracle_detection_ci_95"]; rci = result["realistic_detection_ci_95"]
            m1.metric("Oracle detection rate (true shape known)", f"{result['oracle_detection_rate']*100:.1f}%", help=f"95% CI: {oci[0]*100:.1f}-{oci[1]*100:.1f}%")
            m2.metric("Realistic detection rate (default template)", f"{result['realistic_detection_rate']*100:.1f}%", help=f"95% CI: {rci[0]*100:.1f}-{rci[1]*100:.1f}%")
            st.caption("Framing note: this uploaded profile is an arbitrary real-valued curve, not a claim of biological realism -- this tests template-mismatch sensitivity, not detection of a real neural signal.")

            gap = result['oracle_detection_rate'] - result['realistic_detection_rate']
            if gap > 0.15:
                st.error(f"Large gap ({gap*100:.0f} points) between best-case and realistic detection — "
                         f"this real signal shape is significantly mismatched from our default template. "
                         f"A real device would need either a template library or an adaptive detector.")
            else:
                st.success(f"Small gap ({gap*100:.0f} points) — our default template handles this shape reasonably well.")
    else:
        st.info("Upload an image or NIfTI file above to test detection against a real, non-idealized signal shape.")


with tab_anatomy:
    st.warning(
        "**What this does and does not do:** this measures a REAL physical "
        "distance (scalp surface to cortical surface) from real MRI tissue "
        "segmentation, using actual voxel spacing. It does NOT reconstruct "
        "this person's actual neural firing from their scan ;- the neural "
        "signal strength still comes from the same literature calibrated "
        "model used throughout this project (Barry et al. 2016 + independent "
        "validation), not from anything in the uploaded scan itself. What "
        "changes per-person is the real distance a sensor would have to "
        "reach through  which, as the numbers below show, matters enormously.",
    )

    st.subheader("Real MRI-derived head (SimNIBS \"Ernie\" dataset)")
    st.caption("A real, published, MRI-derived tissue segmentation, used here as a demonstration head "
               "since it's already validated elsewhere in this project. A real uploaded patient scan "
               "in the same NIfTI tissue-segmentation format would work identically.")

    data, voxel_mm = paa.load_tissue_volume("simnibs_data/m2m_ernie/final_tissues.nii.gz")
    brain_mask = (data == 1) | (data == 2)
    counts = brain_mask.sum(axis=(0, 2))
    default_slice = int(counts.argmax())

    slice_index = st.slider("Axial slice (S-index)", int(counts.nonzero()[0].min()),
                              int(counts.nonzero()[0].max()), default_slice)
    angle_deg = st.slider("Measurement angle around the head (degrees)", 0, 350, 180, step=10)

    slice_2d = paa.get_slice(data, axis=1, index=slice_index)
    voxel_inplane = (voxel_mm[0], voxel_mm[2])
    result = paa.measure_standoff_along_ray(slice_2d, angle_deg, voxel_inplane)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(slice_2d.T, cmap="bone", origin="lower")
        if result:
            path = np.array(result["path_coords"])
            ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2, label="Measured path")
        ax.set_title(f"Real head slice, angle={angle_deg}\u00b0")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    with col2:
        if result is None:
            st.error("This ray didn't cleanly cross scalp \u2192 brain (likely hit a sinus, ear canal, "
                      "or eye socket). Try a different angle.")
        else:
            st.metric("Real measured standoff (scalp \u2192 cortex)", f"{result['total_standoff_mm']} mm")
            st.write("**Tissue breakdown along this real path:**")
            for tissue, mm in result["tissue_breakdown_mm"].items():
                st.write(f"- {tissue.replace('_', ' ').title()}: {mm} mm")

    if result:
        st.divider()
        st.subheader(f"What settings would this person's real anatomy require?")
        with st.spinner("Testing real sensor configurations against this measured distance..."):
            settings_results = paa.recommend_settings_for_standoff(result["total_standoff_mm"])

        for signal_label, sens_results in settings_results.items():
            st.write(f"**{signal_label} signal:**")
            cols = st.columns(len(sens_results))
            for col, (sens_label, r) in zip(cols, sens_results.items()):
                with col:
                    if r["achievable"]:
                        st.success(f"{sens_label}\n\n**{r['required_n_averages']}x** averaging needed")
                    else:
                        st.error(f"{sens_label}\n\nNot achievable even at 10,000x averaging")

        st.caption(
            "Real human scalp-to-cortex distance is typically 14-20mm — compare this to the "
            "500\u00b5m-1mm standoff distances tested elsewhere in this project's phantom experiments. "
            "This is the gap between a benchtop/near contact demonstration and a true "
            "non contact wearable device reaching cortical sources."
        )
