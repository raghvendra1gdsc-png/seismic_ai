"""Unit tests for Live Global Seismicity Feed and Local Network Alarm Service."""

import pytest
import time

from src.earthquake.live_feed import GlobalSeismicityFeed, LiveEarthquakeEvent
from src.sensors.alarm import LocalNetworkAlarmService, AlarmBroadcastPayload


def test_global_seismicity_feed():
    """Test real-time or synthetic fallback global earthquake feed fetching."""
    feed = GlobalSeismicityFeed(timeout_seconds=2.0)
    events = feed.fetch_live_events(feed_type="all_day", min_magnitude=3.0, max_events=10)

    assert len(events) > 0
    e0 = events[0]
    assert isinstance(e0, LiveEarthquakeEvent)
    assert e0.magnitude >= 3.0
    assert len(e0.place) > 0
    assert -90.0 <= e0.latitude <= 90.0
    assert -180.0 <= e0.longitude <= 180.0


def test_local_network_alarm_service():
    """Test local network alarm generation, severity levels, and subscriber registry."""
    alarm = LocalNetworkAlarmService()

    # Subscriber registration
    assert alarm.register_subscriber("http://192.168.1.50:8080/buzzer")
    assert not alarm.register_subscriber("http://192.168.1.50:8080/buzzer")  # Duplicate check
    assert len(alarm.subscribers) == 1

    # Generate Critical alarm
    payload_crit = alarm.create_alarm_payload(
        early_pga_g=0.35,
        predicted_drift_pct=1.8,
        lead_time_s=14.5,
        building_name="IIT_Delhi_Academic_Block",
    )
    assert payload_crit.alert_level == "CRITICAL_EVACUATION"
    assert "EVACUATE" in payload_crit.safety_action
    assert payload_crit.audio_chime_type == "siren"

    # Generate Advisory alarm
    payload_adv = alarm.create_alarm_payload(
        early_pga_g=0.02,
        predicted_drift_pct=0.15,
        lead_time_s=8.0,
        building_name="IIT_Delhi_Academic_Block",
    )
    assert payload_adv.alert_level == "ADVISORY"
    assert payload_adv.audio_chime_type == "chime"

    # Unregister subscriber
    assert alarm.unregister_subscriber("http://192.168.1.50:8080/buzzer")
    assert len(alarm.subscribers) == 0
