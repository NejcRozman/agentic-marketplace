"""Tests for heterogeneous provider-group expansion in ExperimentRunner."""

from pathlib import Path

from experiments.runners.experiment_runner import ExperimentRunner


def _build_runner() -> ExperimentRunner:
    return ExperimentRunner(Path("experiments/configs/test.yaml"))


def test_provider_group_ratios_expand_to_expected_mix():
    runner = _build_runner()
    provider_config = {
        "count": 6,
        "config": {"check_interval": 10},
        "behavior": {"auto_bid": True, "auto_execute": True, "auto_complete": True},
        "groups": [
            {
                "name": "a1_react",
                "ratio": 2,
                "behavior": {"architecture": "1", "reasoning_mode": "llm_react"},
            },
            {
                "name": "a2_react",
                "ratio": 1,
                "behavior": {"architecture": "2", "reasoning_mode": "llm_react"},
            },
        ],
    }

    provider_plan = runner._expand_provider_group_configs(provider_config)

    architectures = [entry["group_config"]["behavior"]["architecture"] for entry in provider_plan]
    assert len(provider_plan) == 6
    assert architectures.count("1") == 4
    assert architectures.count("2") == 2


def test_provider_group_overrides_merge_with_shared_defaults():
    runner = _build_runner()
    provider_config = {
        "count": 3,
        "config": {"check_interval": 10, "bidding_base_cost": 0.03},
        "behavior": {"auto_bid": True, "auto_execute": True, "auto_complete": True},
        "groups": [
            {
                "name": "a1_react",
                "ratio": 1,
                "behavior": {"architecture": "1", "reasoning_mode": "llm_react"},
            },
            {
                "name": "a2_react",
                "ratio": 2,
                "config": {"bidding_base_cost": 0.04},
                "behavior": {"architecture": "2", "reasoning_mode": "llm_react"},
            },
        ],
    }

    provider_plan = runner._expand_provider_group_configs(provider_config)
    a2_entries = [entry for entry in provider_plan if entry["group_name"] == "a2_react"]

    assert len(a2_entries) == 2
    assert a2_entries[0]["group_config"]["config"]["check_interval"] == 10
    assert a2_entries[0]["group_config"]["config"]["bidding_base_cost"] == 0.04
    assert a2_entries[0]["group_config"]["behavior"]["auto_bid"] is True
    assert a2_entries[0]["group_config"]["behavior"]["architecture"] == "2"