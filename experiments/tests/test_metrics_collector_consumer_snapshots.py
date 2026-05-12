from pathlib import Path

from experiments.runners.metrics_collector import MetricsCollector


def test_runtime_consumer_auction_populates_quality_and_timing():
    collector = MetricsCollector("exp", Path("tmp-metrics"), Path("tmp-logs"))
    collector.record_consumer_status_snapshot(
        timestamp="2026-05-12 10:04:00",
        status={
            "active_auctions": 0,
            "completed_auctions": 1,
            "failed_auctions": 0,
            "running": True,
            "auctions": [
                {
                    "auction_id": 7,
                    "status": "completed",
                    "ended_at": "2026-05-12T10:01:00",
                    "completed_at": "2026-05-12T10:01:40",
                    "evaluation_started_at": "2026-05-12T10:01:41",
                    "evaluation_completed_at": "2026-05-12T10:01:46",
                    "feedback_submitted_at": "2026-05-12T10:01:48",
                    "service_executed": True,
                    "feedback_submitted": True,
                    "feedback_rating": 88,
                    "quality_rating": 88,
                    "quality_scores": {"accuracy": 90},
                    "quality_explanations": ["Good output"],
                    "quality_method": "consumer_evaluator",
                    "execution_duration_seconds": 40,
                    "evaluation_duration_seconds": 5,
                    "error": None,
                }
            ],
        },
    )

    auction_data = {
        "auction_id": 7,
        "timestamp_created": int(collector._parse_runtime_datetime("2026-05-12T10:00:00").timestamp()),
        "timestamp_ended": int(collector._parse_runtime_datetime("2026-05-12T10:01:00").timestamp()),
        "bids_on_chain": [{"timestamp": int(collector._parse_runtime_datetime("2026-05-12T10:00:10").timestamp())}],
        "service_executed": False,
        "execution_start": None,
        "execution_end": None,
        "execution_duration_seconds": 0,
        "quality_rating": None,
        "quality_method": "unknown",
        "quality_details": {},
        "evaluation_duration_seconds": 0,
        "feedback_submitted": False,
        "feedback_rating": None,
        "timing": {
            "creation_to_first_bid": 0,
            "first_bid_to_close": 0,
            "close_to_execution_start": 0,
            "execution": 0,
            "evaluation": 0,
            "feedback_submission": 0,
            "total_auction_cycle": 0,
        },
        "errors": [],
    }

    runtime_auction = collector._runtime_consumer_auction(7)
    assert runtime_auction is not None

    collector._apply_runtime_consumer_auction(auction_data, runtime_auction)
    collector._calculate_auction_timing(auction_data)

    assert auction_data["service_executed"] is True
    assert auction_data["feedback_submitted"] is True
    assert auction_data["quality_rating"] == 88
    assert auction_data["feedback_rating"] == 88
    assert auction_data["quality_method"] == "consumer_evaluator"
    assert auction_data["quality_details"]["quality_scores"] == {"accuracy": 90}
    assert auction_data["execution_duration_seconds"] == 40
    assert auction_data["evaluation_duration_seconds"] == 5
    assert auction_data["timing"]["execution"] == 40
    assert auction_data["timing"]["evaluation"] == 5
    assert auction_data["timing"]["feedback_submission"] == 2
    assert auction_data["timing"]["total_auction_cycle"] == 107
