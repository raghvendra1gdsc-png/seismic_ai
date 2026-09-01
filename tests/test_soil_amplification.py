"""Unit tests for Geotechnical 1D Soil Column Amplification."""

import pytest
import numpy as np

from src.earthquake.database import GroundMotionDatabase
from src.earthquake.soil_amplification import SoilColumnModel, SoilStratumLayer


def test_soil_column_amplification_and_vs30():
    """Test geotechnical 1D transfer function and Delhi-NCR soil amplification."""
    delhi_soil = SoilColumnModel.delhi_ncr_alluvium()
    assert 200.0 < delhi_soil.vs30 < 350.0  # Medium/Soft alluvial soil range

    himalaya_rock = SoilColumnModel.himalayan_rock_outcrop()
    assert himalaya_rock.vs30 > 800.0  # Rock outcrop

    # Transfer function check
    freqs = np.linspace(0.1, 20.0, 100)
    h_delhi = delhi_soil.transfer_function(freqs)
    # Peak amplification in soft soil must exceed 1.0 (resonance)
    assert np.max(np.abs(h_delhi)) > 1.5

    # Amplify real earthquake record
    db = GroundMotionDatabase()
    rec_rock = db.get_record("Chamoli_1999_Gopeshwar")
    rec_amplified = delhi_soil.amplify_record(rec_rock)

    assert rec_amplified.metadata["site_amplified"]
    assert rec_amplified.metadata["vs30_m_s"] == round(delhi_soil.vs30, 1)
    assert len(rec_amplified.acceleration) == len(rec_rock.acceleration)
