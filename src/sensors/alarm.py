"""Local Network Emergency Alarm Dispatcher & Real-Time Broadcast Service.

Dispatches high-priority early warning alarms upon P-wave onset detection before S-wave arrival:
1. Generates structured JSON emergency broadcast payloads.
2. Dispatches webhooks to local network clients (smart buzzers, building relays, mobile listeners).
3. Provides browser-based Web Audio synthesizer triggers.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import requests
import json


@dataclass
class AlarmBroadcastPayload:
    """Standardized local network earthquake alarm alert packet."""
    alert_id: str
    timestamp_epoch: float
    alert_level: str               # ADVISORY, WARNING, CRITICAL_EVACUATION
    lead_time_seconds: float       # Estimated time before destructive S-wave arrives (e.g. 5 to 25s)
    early_pga_g: float             # Measured P-wave peak acceleration in g
    predicted_max_drift_pct: float # Predicted building peak drift PIDR from AI surrogate
    safety_action: str             # "DROP, COVER, AND HOLD ON", "EVACUATE", "STANDBY"
    target_building: str
    audio_chime_type: str          # "siren", "pulsed_beep", "chime"
    is_drill_test: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp_epoch,
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp_epoch)),
            "alert_level": self.alert_level,
            "estimated_lead_time_s": round(self.lead_time_seconds, 1),
            "measured_p_wave_pga_g": round(self.early_pga_g, 4),
            "predicted_building_drift_pct": round(self.predicted_max_drift_pct, 3),
            "safety_action": self.safety_action,
            "target_building": self.target_building,
            "audio_chime_type": self.audio_chime_type,
            "is_drill_test": self.is_drill_test,
        }


class LocalNetworkAlarmService:
    """Manages active local network alarm subscribers and dispatches alert broadcasts."""

    def __init__(self) -> None:
        self.subscribers: List[str] = []  # List of local IP / webhook URLs
        self.alarm_history: List[AlarmBroadcastPayload] = []

    def register_subscriber(self, webhook_url: str) -> bool:
        """Register a local device/listener endpoint."""
        clean_url = webhook_url.strip()
        if clean_url and clean_url not in self.subscribers:
            self.subscribers.append(clean_url)
            return True
        return False

    def unregister_subscriber(self, webhook_url: str) -> bool:
        """Remove a local listener endpoint."""
        if webhook_url in self.subscribers:
            self.subscribers.remove(webhook_url)
            return True
        return False

    def create_alarm_payload(
        self,
        early_pga_g: float,
        predicted_drift_pct: float,
        lead_time_s: float = 12.0,
        building_name: str = "Structure_01",
        is_test: bool = False,
    ) -> AlarmBroadcastPayload:
        """Construct alarm payload based on AI structural response estimation."""
        # Categorize severity
        if predicted_drift_pct >= 1.5 or early_pga_g >= 0.20:
            level = "CRITICAL_EVACUATION"
            action = "🚨 CRITICAL EARTHQUAKE: EVACUATE / TAKE COVER IMMEDIATELY!"
            chime = "siren"
        elif predicted_drift_pct >= 0.6 or early_pga_g >= 0.05:
            level = "WARNING"
            action = "⚠️ EARTHQUAKE DETECTED: DROP, COVER, AND HOLD ON!"
            chime = "pulsed_beep"
        else:
            level = "ADVISORY"
            action = "ℹ️ MINOR SEISMIC ONSET DETECTED: STANDBY."
            chime = "chime"

        payload = AlarmBroadcastPayload(
            alert_id=f"SEISMIC_ALERT_{int(time.time())}",
            timestamp_epoch=time.time(),
            alert_level=level,
            lead_time_seconds=max(1.0, lead_time_s),
            early_pga_g=early_pga_g,
            predicted_max_drift_pct=predicted_drift_pct,
            safety_action=action,
            target_building=building_name,
            audio_chime_type=chime,
            is_drill_test=is_test,
        )

        self.alarm_history.append(payload)
        return payload

    def broadcast_to_local_network(self, payload: AlarmBroadcastPayload) -> Dict[str, Any]:
        """Dispatch JSON alert payload asynchronously to all local network subscribers."""
        results = {"successful_dispatches": 0, "failed_dispatches": 0, "endpoints": []}
        data_json = payload.to_dict()

        for url in self.subscribers:
            try:
                resp = requests.post(url, json=data_json, timeout=1.0)
                if resp.status_code in (200, 201, 202, 204):
                    results["successful_dispatches"] += 1
                    results["endpoints"].append({"url": url, "status": "SUCCESS"})
                else:
                    results["failed_dispatches"] += 1
                    results["endpoints"].append({"url": url, "status": f"HTTP_{resp.status_code}"})
            except Exception as e:
                results["failed_dispatches"] += 1
                results["endpoints"].append({"url": url, "status": f"ERROR: {str(e)}"})

        return results
