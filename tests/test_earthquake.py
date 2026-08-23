"""Unit tests for earthquake ground-motion processing and response spectra."""

import unittest
import numpy as np

from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.earthquake.processing import baseline_correct, pad_zeros
from src.earthquake.spectra import ResponseSpectrum
from src.earthquake.database import GroundMotionDatabase, generate_stochastic_ground_motion


class TestGroundMotion(unittest.TestCase):
    """Test GroundMotionRecord intensity measures and signal processing."""

    def test_record_intensity_measures(self):
        dt = 0.01
        time = np.arange(0, 10.0, dt)
        # Sinusoidal test accelerogram: a(t) = a0 * sin(2*pi*f * t)
        a0 = 2.0  # 2 m/s^2
        f = 1.0   # 1 Hz (period 1.0s, peak at 0.25s which lands exactly on step 25 with dt=0.01)
        acc = a0 * np.sin(2.0 * np.pi * f * time)

        rec = GroundMotionRecord(name="TestSine", dt=dt, acceleration=acc)

        self.assertAlmostEqual(rec.pga, a0, places=5)
        self.assertAlmostEqual(rec.pga_g, a0 / GRAVITY, places=5)
        self.assertEqual(rec.num_points, len(time))
        self.assertAlmostEqual(rec.duration, time[-1], places=5)

        # Arias intensity for sine wave: Ia = (pi / (2*g)) * integral(a^2 dt)
        # integral(a0^2 * sin^2(w t) dt) over integer cycles is 0.5 * a0^2 * T_tot
        expected_integral = 0.5 * (a0**2) * rec.duration
        expected_ia = (np.pi / (2.0 * GRAVITY)) * expected_integral
        self.assertAlmostEqual(rec.arias_intensity, expected_ia, places=2)

        # Husid vector bounds [0, 1]
        husid = rec.husid_vector
        self.assertAlmostEqual(husid[0], 0.0, places=5)
        self.assertAlmostEqual(husid[-1], 1.0, places=5)
        self.assertTrue(np.all(np.diff(husid) >= -1e-12))  # Monotonic

    def test_scaling_ground_motion(self):
        dt = 0.01
        acc = np.array([0.0, 1.0, -2.0, 1.5, -0.5, 0.0])
        rec = GroundMotionRecord(name="Raw", dt=dt, acceleration=acc)

        scaled = rec.scale_to_pga(target_pga_g=0.5)
        self.assertAlmostEqual(scaled.pga_g, 0.5, places=6)
        self.assertAlmostEqual(scaled.pga, 0.5 * GRAVITY, places=6)

    def test_baseline_correction_and_padding(self):
        dt = 0.01
        time = np.arange(0, 5.0, dt)
        acc = np.sin(time) + 0.5  # Artificial DC offset
        rec = GroundMotionRecord(name="Offset", dt=dt, acceleration=acc)

        corrected = baseline_correct(rec, order=1, zero_mean=True)
        # Mean should now be virtually zero
        self.assertAlmostEqual(np.mean(corrected.acceleration), 0.0, places=7)

        padded = pad_zeros(corrected, leading_seconds=1.0, trailing_seconds=2.0)
        self.assertEqual(padded.num_points, corrected.num_points + 100 + 200)
        self.assertTrue(np.all(padded.acceleration[:100] == 0.0))
        self.assertTrue(np.all(padded.acceleration[-200:] == 0.0))


class TestResponseSpectrum(unittest.TestCase):
    """Test elastic response spectrum generation and asymptotic properties."""

    def test_response_spectrum_asymptotes(self):
        db = GroundMotionDatabase()
        el_centro = db.get_record("El_Centro_1940_NS")

        periods = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
        spec = ResponseSpectrum(record=el_centro, damping_ratio=0.05, periods=periods)

        # 1. At very short periods (T -> 0.02s), Sa approaches PGA
        sa_short = spec.get_sa_g(0.02)
        pga_g = el_centro.pga_g
        # Sa(T=0.02s) is typically within 20% of PGA
        self.assertAlmostEqual(sa_short, pga_g, delta=0.25 * pga_g)

        # 2. Pseudo-velocity Sv = wn * Sd, Pseudo-acceleration Sa = wn^2 * Sd
        for i, tn in enumerate(periods):
            wn = 2.0 * np.pi / tn
            self.assertAlmostEqual(spec.sv[i], wn * spec.sd[i], places=6)
            self.assertAlmostEqual(spec.sa[i], (wn**2) * spec.sd[i], places=6)

    def test_database_initialization(self):
        db = GroundMotionDatabase()
        self.assertGreaterEqual(len(db), 8)
        names = db.list_records()
        self.assertIn("El_Centro_1940_NS", names)
        self.assertIn("Kobe_1995_NS", names)
        self.assertIn("Northridge_1994_Sylmar", names)

        kobe = db["Kobe_1995_NS"]
        self.assertAlmostEqual(kobe.pga_g, 0.834, delta=0.01)


if __name__ == "__main__":
    unittest.main()
