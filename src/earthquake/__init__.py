"""Earthquake ground-motion processing and spectral analysis module."""

from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.earthquake.processing import baseline_correct, pad_zeros
from src.earthquake.spectra import ResponseSpectrum
from src.earthquake.database import GroundMotionDatabase, generate_stochastic_ground_motion
from src.earthquake.soil_amplification import SoilColumnModel, SoilStratumLayer

EarthquakeDatabase = GroundMotionDatabase

__all__ = [
    "GroundMotionRecord",
    "GRAVITY",
    "baseline_correct",
    "pad_zeros",
    "ResponseSpectrum",
    "GroundMotionDatabase",
    "EarthquakeDatabase",
    "generate_stochastic_ground_motion",
    "SoilColumnModel",
    "SoilStratumLayer",
]
