from config.simulation_config import (
    SIMULATION_DAYS,
    STATISTICS_INTERVAL_DAYS,
    WARM_UP_DAYS,
)
import main


def test_python_run_configuration_exports_config_values():
    assert main.SIMULATION_DAYS == SIMULATION_DAYS
    assert main.STATISTICS_INTERVAL_DAYS == STATISTICS_INTERVAL_DAYS
    assert main.WARM_UP_DAYS == WARM_UP_DAYS
