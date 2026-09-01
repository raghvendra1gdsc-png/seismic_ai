"""Geotechnical 1D Equivalent Linear Soil Stratum Amplification.

Computes frequency-dependent soil column transfer function H(omega) and converts
rock outcrop ground motions into surface ground motions for deep alluvial basins
(e.g., Indo-Gangetic Plain, Delhi-NCR, Kachchh Basin) versus Himalayan rock outcrops.

References:
- Kramer, S. L. (1996). Geotechnical Earthquake Engineering. Prentice Hall.
- Schnabel, P. B., Lysmer, J., & Seed, H. B. (1972). SHAKE: A computer program for
  earthquake response analysis of horizontally layered sites.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.earthquake.record import GroundMotionRecord


@dataclass
class SoilStratumLayer:
    """Individual horizontal soil stratum layer."""
    layer_name: str
    thickness_m: float           # Layer thickness H_i in meters
    shear_wave_velocity_m_s: float # Shear wave velocity V_s,i in m/s
    density_kg_m3: float = 1900.0  # Mass density rho in kg/m^3
    damping_ratio: float = 0.05    # Soil material damping xi_s


class SoilColumnModel:
    """Multi-layer 1D viscoelastic geotechnical soil column.

    Parameters
    ----------
    layers : List[SoilStratumLayer]
        Stratified soil layers from surface to bedrock.
    bedrock_vs_m_s : float, default=1500.0
        Bedrock half-space shear wave velocity (rock outcrop).
    bedrock_density_kg_m3 : float, default=2400.0
        Bedrock half-space density in kg/m^3.
    """

    def __init__(
        self,
        layers: List[SoilStratumLayer],
        bedrock_vs_m_s: float = 1500.0,
        bedrock_density_kg_m3: float = 2400.0,
        site_name: str = "Generic Soil Column",
    ) -> None:
        self.layers = layers
        self.bedrock_vs = float(bedrock_vs_m_s)
        self.bedrock_density = float(bedrock_density_kg_m3)
        self.site_name = site_name

        self.total_thickness = float(np.sum([layer.thickness_m for layer in layers]))
        # Compute time-averaged shear wave velocity in top 30m (Vs30)
        self.vs30 = self._compute_vs30()

    def _compute_vs30(self) -> float:
        """Compute standard geotechnical V_s30 (m/s)."""
        d_acc = 0.0
        t_acc = 0.0
        for lay in self.layers:
            d_lay = min(lay.thickness_m, max(0.0, 30.0 - d_acc))
            if d_lay > 0:
                t_acc += d_lay / max(lay.shear_wave_velocity_m_s, 1e-3)
                d_acc += d_lay
            if d_acc >= 30.0:
                break
        if d_acc < 30.0:
            # Remainder in bedrock
            t_acc += (30.0 - d_acc) / self.bedrock_vs
        return 30.0 / max(t_acc, 1e-4)

    def transfer_function(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Compute complex transfer function H(omega) for single equivalent stratum."""
        freqs = np.asarray(frequencies_hz, dtype=np.float64)
        omega = 2.0 * np.pi * freqs

        # Equivalent single-layer approximation for stratum:
        h_tot = self.total_thickness
        vs_avg = self.vs30
        xi_avg = float(np.mean([l.damping_ratio for l in self.layers])) if self.layers else 0.05
        rho_avg = float(np.mean([l.density_kg_m3 for l in self.layers])) if self.layers else 1900.0

        # Complex wave number k* = omega / (V_s * sqrt(1 + 2*i*xi))
        vs_star = vs_avg * np.sqrt(1.0 + 2.0j * xi_avg)
        k_star = omega / np.maximum(vs_star, 1e-4)

        # Impedance ratio alpha_z = (rho_soil * Vs_soil) / (rho_rock * Vs_rock)
        alpha_z = (rho_avg * vs_avg) / (self.bedrock_density * self.bedrock_vs)

        # Classical transfer function for soil layer on elastic rock half-space:
        # H(omega) = 1 / (cos(k* H) + i * alpha_z * sin(k* H))
        denom = np.cos(k_star * h_tot) + 1.0j * alpha_z * np.sin(k_star * h_tot)
        # Avoid division by zero
        denom = np.where(np.abs(denom) < 1e-6, 1e-6 + 0j, denom)
        h_omega = 1.0 / denom

        return h_omega

    def amplify_record(self, rock_record: GroundMotionRecord) -> GroundMotionRecord:
        """Transform rock outcrop acceleration record to surface ground motion via FFT."""
        ag = rock_record.acceleration
        dt = rock_record.dt
        n = len(ag)

        # FFT
        fft_ag = np.fft.rfft(ag)
        freqs = np.fft.rfftfreq(n, d=dt)

        # Apply transfer function H(omega)
        h_omega = self.transfer_function(freqs)
        fft_surface = fft_ag * h_omega

        # Inverse FFT
        ag_surface = np.fft.irfft(fft_surface, n=n)

        # Package as new GroundMotionRecord
        meta = dict(rock_record.metadata)
        meta["site_amplified"] = True
        meta["site_name"] = self.site_name
        meta["vs30_m_s"] = round(self.vs30, 1)

        return GroundMotionRecord(
            name=f"{rock_record.name}_Surface_{self.site_name.replace(' ', '_')}",
            dt=dt,
            acceleration=ag_surface,
            metadata=meta,
        )

    @classmethod
    def delhi_ncr_alluvium(cls) -> "SoilColumnModel":
        """Pre-configured Deep Indo-Gangetic Alluvium profile for Delhi-NCR (IS 1893 Type III Soft/Medium)."""
        layers = [
            SoilStratumLayer("Surface Silty Sand", thickness_m=5.0, shear_wave_velocity_m_s=180.0, damping_ratio=0.06),
            SoilStratumLayer("Medium Dense Alluvial Silt", thickness_m=15.0, shear_wave_velocity_m_s=250.0, damping_ratio=0.05),
            SoilStratumLayer("Dense Sandy Gravel", thickness_m=20.0, shear_wave_velocity_m_s=380.0, damping_ratio=0.04),
        ]
        return cls(layers, bedrock_vs_m_s=1600.0, site_name="Delhi-NCR Indo-Gangetic Deep Alluvium")

    @classmethod
    def himalayan_rock_outcrop(cls) -> "SoilColumnModel":
        """Pre-configured Himalayan Competent Quartzite Rock Profile (IS 1893 Type I Rock)."""
        layers = [
            SoilStratumLayer("Weathered Quartzite Top Layer", thickness_m=3.0, shear_wave_velocity_m_s=900.0, damping_ratio=0.02),
        ]
        return cls(layers, bedrock_vs_m_s=2200.0, site_name="Himalayan Garhwal Quartzite Rock")
