import csv

import pytest

from resilience_kpi_calculator import calculate_resilience_kpi


def _write_periods(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["PeriodIndex", "StartDay", "EndDay", "AverageTransportTime"]
        )
        writer.writerows(rows)
        writer.writerow(["", "", "OverallMean", "0.00"])


def test_calculate_resilience_kpi(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    disruption_path = tmp_path / "disruption.csv"
    output_path = tmp_path / "ATT_Resilience_By_Period.csv"
    _write_periods(
        baseline_path,
        [
            [1, 1, 10, 4.0],
            [2, 11, 20, 5.0],
        ],
    )
    _write_periods(
        disruption_path,
        [
            [1, 1, 10, 5.0],
            [2, 11, 20, 4.0],
        ],
    )

    rows = calculate_resilience_kpi(
        baseline_path,
        disruption_path,
        output_path,
    )

    assert rows[0]["Q"] == pytest.approx(0.8)
    assert rows[0]["PeriodResilienceLoss"] == pytest.approx(2.0)
    assert rows[0]["CumulativeResilienceLoss"] == pytest.approx(2.0)
    assert rows[1]["Q"] == 1.0
    assert rows[1]["PeriodResilienceLoss"] == 0.0
    assert rows[1]["CumulativeResilienceLoss"] == pytest.approx(2.0)

    with output_path.open(newline="", encoding="utf-8") as stream:
        output_rows = list(csv.DictReader(stream))
    assert output_rows[0]["Q"] == "0.800000"
    assert output_rows[1]["CumulativeResilienceLoss"] == "2.000000"


def test_calculate_resilience_kpi_rejects_mismatched_periods(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    disruption_path = tmp_path / "disruption.csv"
    _write_periods(baseline_path, [[1, 1, 10, 4.0]])
    _write_periods(disruption_path, [[1, 1, 9, 5.0]])

    with pytest.raises(ValueError, match="different start or end days"):
        calculate_resilience_kpi(
            baseline_path,
            disruption_path,
            tmp_path / "output.csv",
        )
