from typing import Dict

from app.service.result_service import (
    load_ranking,
    load_top_tests,
    load_metrics
)


def load_full_results(experiment_id: int) -> Dict:
    """
    Loads all experiment outputs together.
    Useful for reports / dashboards / exports.
    """
    return {
        "experiment_id": experiment_id,
        "ranking": load_ranking(experiment_id),
        "top_tests": load_top_tests(experiment_id),
        "metrics": load_metrics(experiment_id),
    }
