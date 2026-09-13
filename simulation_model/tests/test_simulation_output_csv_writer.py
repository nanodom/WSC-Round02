import datetime as dt
import csv

import pytest

from maritime_data_context import Demand, MaritimeDataContext, MaritimeDataInitializer, Port, Shipment
from simulation_model import Model
from simulation_output_csv_writer import (
    write_all,
    write_att_by_period,
)


def test_writer_creates_dashboard_csv_files(tmp_path):
    sim = Model(MaritimeDataInitializer.create(), seed=1)
    sim.run(duration=dt.timedelta(days=1))

    write_all(sim, tmp_path)
    write_att_by_period(tmp_path, [(1, 1, 1.25)])
    expected = {
        "Average_In_Transit_TEU_By_OD.csv",
        "Cumulative_Completed_TEU_By_OD.csv",
        "ATT_By_Statistics_Interval.csv",
        "Port_Waiting_Statistics.csv",
        "Service_Route_Utilization.csv",
        "Average_Origin_Waiting_TEU_By_OD.csv",
        "Average_Vessel_State_Counts.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    with (tmp_path / "ATT_By_Statistics_Interval.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [
        "PeriodIndex",
        "StartDay",
        "EndDay",
        "AverageTransportTime",
    ]

def test_teu_weighted_average_transport_time_includes_older_backlog():
    context = MaritimeDataContext()
    origin = Port(name="Origin")
    destination = Port(name="Destination")
    context.ports.extend([origin, destination])
    demand = Demand(origin, destination, annual_teus=0)
    context.demands.append(demand)
    sim = Model(context, seed=1)

    older = Shipment(
        index=1,
        teu_size=100,
        demand=demand,
        generated_time=sim.clock_time,
    )
    completed_before_period = Shipment(
        index=2,
        teu_size=1000,
        demand=demand,
        generated_time=sim.clock_time,
    )
    completed_before_period.completion_time = (
        sim.clock_time + dt.timedelta(hours=6)
    )
    completed = Shipment(
        index=3,
        teu_size=2,
        demand=demand,
        generated_time=sim.clock_time + dt.timedelta(days=1),
    )
    completed.completion_time = sim.clock_time + dt.timedelta(days=3)
    unfinished = Shipment(
        index=4,
        teu_size=1,
        demand=demand,
        generated_time=sim.clock_time + dt.timedelta(days=2),
    )
    demand.shipments.extend(
        [older, completed_before_period, completed, unfinished]
    )

    sim.run(duration=dt.timedelta(days=4))

    # The 100-TEU older backlog contributes four days of age even though it
    # was generated before this period. The shipment completed before this
    # period is excluded. The other two shipments each contribute two days.
    expected_hours = (100 * 96 + 2 * 48 + 1 * 48) / 103
    actual_hours = sim.get_teu_weighted_average_transport_time_hours(
        sim.simulation_start_time + dt.timedelta(days=0.5),
        sim.clock_time,
    )
    assert actual_hours == pytest.approx(expected_hours)


def test_od_flow_status_reports_period_flow_and_backlog_by_demand():
    context = MaritimeDataContext()
    origin = Port(name="Origin")
    destination = Port(name="Destination")
    other_destination = Port(name="Other")
    context.ports.extend([origin, destination, other_destination])
    demand = Demand(origin, destination, annual_teus=365)
    other_demand = Demand(origin, other_destination, annual_teus=0)
    context.demands.extend([demand, other_demand])
    sim = Model(context, seed=1)
    start = sim.simulation_start_time

    completed_before_period = Shipment(
        index=1,
        teu_size=10,
        demand=demand,
        generated_time=start,
    )
    completed_before_period.completion_time = start + dt.timedelta(days=1)
    completed_in_period = Shipment(
        index=2,
        teu_size=20,
        demand=demand,
        generated_time=start + dt.timedelta(days=1.5),
    )
    completed_in_period.completion_time = start + dt.timedelta(days=3)
    older_backlog = Shipment(
        index=3,
        teu_size=30,
        demand=demand,
        generated_time=start,
    )
    new_backlog = Shipment(
        index=4,
        teu_size=40,
        demand=demand,
        generated_time=start + dt.timedelta(days=2),
    )
    demand.shipments.extend([
        completed_before_period,
        completed_in_period,
        older_backlog,
        new_backlog,
    ])

    period_start = start + dt.timedelta(days=1)
    period_end = start + dt.timedelta(days=4)
    rows = sim.get_od_flow_status_for_period(period_start, period_end)
    by_od = {(row["origin"], row["destination"]): row for row in rows}
    status = by_od[("Origin", "Destination")]

    assert status["annual_teus"] == 365
    assert status["generated_teu"] == 60
    assert status["completed_teu"] == 20
    assert status["cumulative_completed_teu"] == 30
    assert status["backlog_teu"] == 70
    assert status["average_completed_att_hours"] == 36
    assert status["average_backlog_age_hours"] == pytest.approx(
        (30 * 96 + 40 * 48) / 70
    )
    assert by_od[("Origin", "Other")]["backlog_teu"] == 0


def test_sandbox_warmup_resets_activity_statistics():
    context = MaritimeDataContext()
    origin = Port(name="Origin")
    destination = Port(name="Destination")
    context.ports.extend([origin, destination])
    demand = Demand(origin, destination, annual_teus=0)
    context.demands.append(demand)
    sim = Model(context, seed=1)

    waiting_counter = (
        sim.shipment_waiting_for_loading_at_origin_port
        .hc_teus_by_demand[demand]
    )
    waiting_counter.observe_change(10)
    waiting_counter.observe_change(-2)

    sim.schedule(lambda: None, delay=dt.timedelta(days=2))
    sim.warmup(period=dt.timedelta(days=2))

    assert waiting_counter.average_count == 8
    assert waiting_counter.total_decrement == 0
