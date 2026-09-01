"""Earthquake ground-motion processing, spectral analysis, and real-time global feeds."""

from src.earthquake.record import GroundMotionRecord, GRAVITY
from src.earthquake.processing import baseline_correct, pad_zeros
from src.earthquake.spectra import ResponseSpectrum
from src.earthquake.database import GroundMotionDatabase, generate_stochastic_ground_motion
from src.earthquake.soil_amplification import SoilColumnModel, SoilStratumLayer
from src.earthquake.live_feed import GlobalSeismicityFeed, LiveEarthquakeEvent

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
    "GlobalSeismicityFeed",
    "LiveEarthquakeEvent",
]
