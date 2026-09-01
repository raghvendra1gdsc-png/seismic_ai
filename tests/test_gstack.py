"""Unit tests for gstack integration and workflow automation."""

import pytest
import os
import json
from scripts.gstack import load_gstack_config, run_status, run_review, run_office_hours


def test_gstack_config_validity():
    """Verify that .gstack/config.json exists and has all required personas and quality gates."""
    config = load_gstack_config()
    assert config["name"] == "gstack-seismic-ai"
    assert "ceo" in config["personas"]
    assert "eng_manager" in config["personas"]
    assert "researcher" in config["personas"]
    assert "qa_lead" in config["personas"]
    assert "security_officer" in config["personas"]
    assert config["quality_gates"]["min_test_count"] >= 74


def test_gstack_roles_and_workflows_exist():
    """Verify that all markdown role definitions and workflow documents exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    roles_dir = os.path.join(base_dir, ".gstack", "roles")
    workflows_dir = os.path.join(base_dir, ".gstack", "workflows")

    assert os.path.exists(os.path.join(roles_dir, "ceo.md"))
    assert os.path.exists(os.path.join(roles_dir, "eng_manager.md"))
    assert os.path.exists(os.path.join(roles_dir, "researcher.md"))
    assert os.path.exists(os.path.join(roles_dir, "qa_lead.md"))
    assert os.path.exists(os.path.join(roles_dir, "security_officer.md"))

    assert os.path.exists(os.path.join(workflows_dir, "review.md"))
    assert os.path.exists(os.path.join(workflows_dir, "qa.md"))
    assert os.path.exists(os.path.join(workflows_dir, "ship.md"))
    assert os.path.exists(os.path.join(workflows_dir, "office-hours.md"))


def test_gstack_cli_functions():
    """Verify that gstack status, review, and office-hours execute cleanly."""
    assert run_status() == 0
    assert run_review() == 0
    assert run_office_hours() == 0
