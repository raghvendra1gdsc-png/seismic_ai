"""Populates raw and processed earthquake record files and metadata."""

import os
import json
import numpy as np

from src.earthquake.database import GroundMotionDatabase
from src.earthquake.processing import baseline_correct
from src.earthquake.spectra import ResponseSpectrum


def export_earthquake_data():
    raw_dir = "data/raw/earthquake_records"
    proc_dir = "data/processed/earthquake_records"
    meta_dir = "data/raw/metadata"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    db = GroundMotionDatabase()
    records = db.get_all_records()

    all_metadata = []

    for rec in records:
        # 1. Raw accelerogram export
        raw_csv_path = os.path.join(raw_dir, f"{rec.name}.csv")
        t = rec.time
        acc = rec.acceleration
        acc_g = rec.acceleration_g

        with open(raw_csv_path, "w") as f:
            f.write("time_s,accel_m_s2,accel_g\n")
            for ti, ai, agi in zip(t, acc, acc_g):
                f.write(f"{ti:.4f},{ai:.6f},{agi:.6f}\n")

        # 2. Processed baseline-corrected record & response spectrum
        corrected = baseline_correct(rec)
        spec = ResponseSpectrum(record=corrected, damping_ratio=0.05)

        proc_csv_path = os.path.join(proc_dir, f"{rec.name}_processed.csv")
        with open(proc_csv_path, "w") as f:
            f.write("time_s,accel_corrected_m_s2,vel_m_s,disp_m\n")
            v = corrected.velocity
            d = corrected.displacement
            for ti, ai, vi, di in zip(corrected.time, corrected.acceleration, v, d):
                f.write(f"{ti:.4f},{ai:.6f},{vi:.6f},{di:.6f}\n")

        spec_csv_path = os.path.join(proc_dir, f"{rec.name}_spectrum_5pct.csv")
        with open(spec_csv_path, "w") as f:
            f.write("period_s,Sa_m_s2,Sa_g,Sv_m_s,Sd_m\n")
            for pi, sa, sag, sv, sd in zip(spec.periods, spec.sa, spec.sa_g, spec.sv, spec.sd):
                f.write(f"{pi:.4f},{sa:.6f},{sag:.6f},{sv:.6f},{sd:.6f}\n")

        # 3. Compile metadata
        rec_meta = rec.to_dict()
        all_metadata.append(rec_meta)

    with open(os.path.join(meta_dir, "earthquake_suite_metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"[+] Successfully exported {len(records)} ground motion records to data/raw and data/processed.")


if __name__ == "__main__":
    export_earthquake_data()
