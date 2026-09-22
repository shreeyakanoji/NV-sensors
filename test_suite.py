import unittest
import numpy as np

import axon_field_model as afm
from phantom_experiment import build_array_and_template, clean_trace
from detection_pipeline import matched_filter_statistic, monte_carlo_detection_test
from realistic_noise_model import generate_pink_noise, realistic_sensor_noise, jittered_pulse_params



class TestAxonFieldPhysics(unittest.TestCase):

    def test_ampere_law_known_case(self):
       
        I_test = 1.0  # 1 Amp
        r_test = 1e-3  # 1mm
        expected_T = afm.MU0 * I_test / (2 * np.pi * r_test)
        result_pT = afm.axon_field_pT(I_test, r_test)
        self.assertAlmostEqual(result_pT, expected_T * 1e12, places=3)

    def test_field_falls_off_as_one_over_r(self):
       
        I_test = 1e-6
        B1 = afm.axon_field_pT(I_test, 1e-4)
        B2 = afm.axon_field_pT(I_test, 2e-4)
        self.assertAlmostEqual(B1 / B2, 2.0, places=6)

    def test_calibration_reproduces_input_measurement(self):

      
        r_m = afm.R_GIANT_MEASURED_UM * 1e-6
        recovered_pT = afm.axon_field_pT(afm.I_GIANT_A, r_m)
        self.assertAlmostEqual(recovered_pT, afm.B_GIANT_MEASURED_PT, places=6)

    def test_mammalian_current_is_smaller_than_giant(self):

      
        self.assertLess(afm.I_MAMMALIAN_A, afm.I_GIANT_A)
        self.assertAlmostEqual(afm.I_MAMMALIAN_A / afm.I_GIANT_A,
                                 afm.D_MAMMALIAN_AXON_UM / afm.D_GIANT_AXON_UM, places=6)

    def test_zero_current_gives_zero_field(self):
        self.assertEqual(afm.axon_field_pT(0.0, 1e-4), 0.0)

    def test_noise_floor_scales_correctly_with_averaging(self):

      
        n1 = afm.noise_floor_pT(1.0, 1000, n_averages=1)
        n100 = afm.noise_floor_pT(1.0, 1000, n_averages=100)
        self.assertAlmostEqual(n1 / n100, 10.0, places=6)


class TestPhantomExperiment(unittest.TestCase):

    def test_signal_peaks_at_expected_time(self):
      
        z_sensor = 1e-3
        time_points = np.linspace(-2e-3, 2e-3, 500)
        trace = clean_trace(1e-6, z_sensor, time_points, 500e-6)
        peak_time = time_points[np.argmax(trace)]
        expected_time = z_sensor / 2.0  # CONDUCTION_VELOCITY_M_S = 2.0
        self.assertAlmostEqual(peak_time, expected_time, delta=0.02e-3)  # within one time-step tolerance

    def test_array_has_correct_shape(self):
        sensor_z, time_points, templates = build_array_and_template(1e-6, 500e-6, n_sensors=7)
        self.assertEqual(len(sensor_z), 7)
        self.assertEqual(templates.shape[0], 7)
        self.assertEqual(templates.shape[1], len(time_points))

    def test_closer_standoff_gives_larger_signal(self):
        _, _, far = build_array_and_template(1e-6, 1000e-6)
        _, _, near = build_array_and_template(1e-6, 100e-6)
        self.assertGreater(near.max(), far.max())


class TestMatchedFilter(unittest.TestCase):

    def test_perfect_match_maximizes_statistic(self):
       
       
        _, _, template = build_array_and_template(1e-6, 500e-6)
        perfect_stat, _ = matched_filter_statistic(template, template)

        shifted_template = np.roll(template, 20, axis=1)  # time-shifted mismatch
        mismatched_stat, _ = matched_filter_statistic(shifted_template, template)

        self.assertGreater(perfect_stat, mismatched_stat)

    def test_pure_noise_gives_near_zero_expected_statistic(self):
        
        _, _, template = build_array_and_template(1e-6, 500e-6)
        rng = np.random.default_rng(0)
        stats = []
        for _ in range(500):
            noise = rng.normal(0, 1.0, template.shape)
            stat, _ = matched_filter_statistic(noise, template)
            stats.append(stat)
        self.assertAlmostEqual(np.mean(stats), 0.0, delta=0.5)


class TestRealisticNoiseModel(unittest.TestCase):

    def test_pink_noise_has_1_over_f_spectrum(self):

      
        rng = np.random.default_rng(0)
        n_samples = 4096
        pink = generate_pink_noise(n_samples, rng)
        freqs = np.fft.rfftfreq(n_samples)[1:]
        power = np.abs(np.fft.rfft(pink))[1:] ** 2
        valid = freqs > 0
        slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid] + 1e-15), 1)

      
        self.assertLess(slope, -0.5)
        self.assertGreater(slope, -1.8)

    def test_common_mode_noise_is_correlated_across_sensors(self):
        rng = np.random.default_rng(0)


      
        noise = realistic_sensor_noise((5, 300), 1.0, rng, pink_fraction=0.5,
                                         common_mode_fraction=1.0)
        for i in range(1, 5):
            corr = np.corrcoef(noise[0], noise[i])[0, 1]
            self.assertGreater(corr, 0.99)

    def test_independent_noise_when_no_common_mode(self):
        rng = np.random.default_rng(0)
        noise = realistic_sensor_noise((5, 300), 1.0, rng, pink_fraction=0.5,
                                         common_mode_fraction=0.0)
        corr = np.corrcoef(noise[0], noise[1])[0, 1]
        self.assertLess(abs(corr), 0.3) 
      
    def test_jitter_stays_within_stated_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            v, w, a = jittered_pulse_params(rng, velocity_jitter_frac=0.20,
                                              width_jitter_frac=0.20, amplitude_jitter_frac=0.30)
            self.assertTrue(0.80 <= v <= 1.20)
            self.assertTrue(0.80 <= w <= 1.20)
            self.assertTrue(0.70 <= a <= 1.30)




class TestScientificValidation(unittest.TestCase):

    def test_giant_axon_current_is_physiologically_plausible(self):
        
        self.assertGreater(afm.I_GIANT_A, 0.1e-6)
        self.assertLess(afm.I_GIANT_A, 10e-6)

    def test_independent_literature_agreement_within_tolerance(self):
        
        result = afm.validate_against_independent_literature()
        self.assertGreater(result["agreement_pct"], 60.0,
                            "Model diverged significantly from independent literature check -- "
                            "investigate before trusting downstream results")

    def test_detection_rate_regression(self):
       
        result = monte_carlo_detection_test(afm.I_MAMMALIAN_A, 500e-6, "regression check",
                                              n_trials=300, seed=123)
        self.assertGreater(result["detection_rate"], 0.90,
                            "Detection rate for the validated hard case dropped below "
                            "the previously-established baseline -- check for regressions")


if __name__ == "__main__":
    print("=" * 70)
    print("Running full test suite: software correctness + scientific validation")
    print("=" * 70)
    unittest.main(verbosity=2)
