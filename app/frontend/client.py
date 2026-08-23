"""HTTP and WebSocket client for communicating with the FastAPI backend."""

from typing import Dict, Any, List, Optional
import requests


class BackendClient:
    """Client for the Seismic-AI FastAPI backend running on localhost."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        """Check backend health status."""
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def list_earthquakes(self) -> Dict[str, Any]:
        """Get list of earthquake records."""
        resp = requests.get(f"{self.base_url}/earthquakes", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_earthquake_details(self, name: str) -> Dict[str, Any]:
        """Get time series and response spectra for a record."""
        resp = requests.get(f"{self.base_url}/earthquake/{name}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """Get benchmark buildings catalog."""
        resp = requests.get(f"{self.base_url}/benchmarks", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def predict(
        self,
        building_dict: Dict[str, Any],
        earthquake_name: str,
        model_name: str = "GradientBoosting",
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Call POST /predict on backend."""
        payload = {
            "building": building_dict,
            "earthquake_name": earthquake_name,
            "model_name": model_name,
            "scale_factor": scale_factor,
        }
        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def simulate(
        self,
        building_dict: Dict[str, Any],
        earthquake_name: str,
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Call POST /simulate on backend."""
        payload = {
            "building": building_dict,
            "earthquake_name": earthquake_name,
            "scale_factor": scale_factor,
        }
        resp = requests.post(f"{self.base_url}/simulate", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def compare_is1893(
        self,
        building_dict: Dict[str, Any],
        earthquake_name: str,
        zone: str = "Zone IV",
        soil_type: str = "Type II (Medium)",
        model_name: str = "GradientBoosting",
    ) -> Dict[str, Any]:
        """Call POST /is1893/compare on backend."""
        payload = {
            "building": building_dict,
            "earthquake_name": earthquake_name,
            "zone": zone,
            "soil_type": soil_type,
            "model_name": model_name,
        }
        resp = requests.post(f"{self.base_url}/is1893/compare", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
