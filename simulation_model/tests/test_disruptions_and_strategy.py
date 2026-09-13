import datetime as dt

from maritime_data_context import (
    Berth,
    Booking,
    Demand,
    DisruptionPlan,
    Leg,
    MaritimeDataContext,
    Port,
    Segment,
    ServiceRoute,
    Shipment,
    Vessel,
)
from simulation_model import DisruptionManager
import simulation_model.shipment_waiting_for_loading_at_origin_port as origin_activity_module
import simulation_model.berth_idle as berth_idle_module
import simulation_model.vessel_queuing_for_berth as vessel_queue_module
from simulation_model.disruption_status import is_disruption_active
from simulation_model.shipment_waiting_for_loading_at_origin_port import (
    ShipmentWaitingForLoadingAtOriginPort,
)
from simulation_model.berth_idle import BerthIdle
from simulation_model.vessel_being_served import VesselBeingServed
from simulation_model.vessel_queuing_for_berth import VesselQueuingForBerth
from response_strategies import DefaultStrategy, UserStrategy
from scenario_builders.disruption_scenario import (
    CLOSED_PORTS,
    CONGESTED_LEGS,
    _add_closed_port,
    create_with_disruption,
)
from o2des.core import Sandbox


def test_disruption_manager_applies_and_restores_plan():
    context = MaritimeDataContext()
    origin = Port(name="Origin")
    destination = Port(name="Destination")
    leg = Leg(origin, destination, 100)
    berth = Berth(index=0, port=destination)
    context.disruption_plans.extend(
        [
            DisruptionPlan(leg, start_offset_days=1, duration_days=2, multiplier=3),
            DisruptionPlan(
                target_berth=berth,
                start_offset_days=1,
                duration_days=2,
                close_berth=True,
            ),
        ]
    )

    root = Sandbox(seed=1)
    root.add_child(DisruptionManager(context, root.clock_time, seed=2))

    root.run(duration=dt.timedelta(days=1))
    assert leg.sailing_time_multiplier == 3
    assert berth.is_available is False

    root.run(duration=dt.timedelta(days=2))
    assert leg.sailing_time_multiplier == 1
    assert berth.is_available is True


def test_disruption_status_uses_start_inclusive_end_exclusive_interval():
    context = MaritimeDataContext()
    context.disruption_plans.append(
        DisruptionPlan(
            start_offset_days=50,
            duration_days=10,
        )
    )

    assert not is_disruption_active(
        context,
        dt.datetime.min + dt.timedelta(days=49),
    )
    assert is_disruption_active(
        context,
        dt.datetime.min + dt.timedelta(days=50),
    )
    assert not is_disruption_active(
        context,
        dt.datetime.min + dt.timedelta(days=60),
    )


def test_enabled_strategy_uses_default_strategy_before_disruption(monkeypatch):
    monkeypatch.setattr(origin_activity_module, "ENABLE_STRATEGY", True)
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B")}
    context.ports.extend(ports.values())
    baseline_route = _add_route(
        context,
        "BASELINE",
        [("A", "B", 1), ("B", "A", 1)],
    )
    activity = ShipmentWaitingForLoadingAtOriginPort(context)
    current_offset_days = (
        activity.clock_time - dt.datetime.min
    ).total_seconds() / (24 * 60 * 60)
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=baseline_route.segments[0].associated_leg,
            start_offset_days=current_offset_days + 50,
            duration_days=10,
            multiplier=3,
        )
    )
    demand = Demand(origin_port=ports["A"], destination_port=ports["B"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)
    default_calls = []
    original_assign = DefaultStrategy.assign_associated_bookings

    def record_default_call(*args, **kwargs):
        default_calls.append((args, kwargs))
        return original_assign(*args, **kwargs)

    monkeypatch.setattr(
        DefaultStrategy, "assign_associated_bookings", staticmethod(record_default_call)
    )

    assert activity._assign_associated_bookings(shipment)
    assert len(default_calls) == 1
    assert shipment.associated_bookings[0].service_route is baseline_route


def test_disruption_scenario_offsets_start_days_by_warm_up():
    warm_up_days = 90
    context = create_with_disruption(warm_up_days=warm_up_days)

    assert context.disruption_plans
    assert {
        plan.start_offset_days for plan in context.disruption_plans
    } == {
        warm_up_days + disruption[2] for disruption in CONGESTED_LEGS
    } | {
        warm_up_days + disruption[1] for disruption in CLOSED_PORTS
    }


def test_disruption_scenario_applies_congestion_to_all_matching_legs():
    context = create_with_disruption(warm_up_days=90)
    disrupted_legs = {
        plan.target_leg for plan in context.disruption_plans
        if plan.target_leg is not None
    }

    for departure, arrival, *_ in CONGESTED_LEGS:
        matching_legs = {
            leg for leg in context.legs
            if leg.departure_port.name == departure
            and leg.arrival_port.name == arrival
        }
        assert matching_legs
        assert matching_legs <= disrupted_legs


def test_disruption_scenario_has_reroutable_staggered_disruptions():
    assert CONGESTED_LEGS == [
        ("Colombo", "New Jersey", 40.0, 60.0, 5.0),
        ("Shanghai", "Kaohsiung", 140.0, 60.0, 5.0),
        ("Qingdao", "Busan", 215.0, 25.0, 5.0),
    ]
    assert CLOSED_PORTS == [
        ("Piraeus", 260.0, 14.0),
        ("Tianjin", 320.0, 7.0),
    ]


def test_closed_port_disruption_applies_to_every_berth():
    context = MaritimeDataContext()
    port = Port(name="Hub")
    port.berths.extend([
        Berth(index=0, port=port),
        Berth(index=1, port=port),
    ])
    context.ports.append(port)

    _add_closed_port(
        context,
        "Hub",
        start_offset_days=10,
        duration_days=5,
    )

    assert len(context.disruption_plans) == 2
    assert {
        plan.target_berth for plan in context.disruption_plans
    } == set(port.berths)
    assert all(plan.close_berth for plan in context.disruption_plans)


def test_disabled_strategy_uses_fifo_and_skips_strategy_calls(monkeypatch):
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B")}
    context.ports.extend(ports.values())
    route = _add_route(context, "ROUTE", [("A", "B", 1), ("B", "A", 1)])
    berth = Berth(index=0, port=ports["B"])
    first_vessel = Vessel(index=1, assigned_service_route=route)
    second_vessel = Vessel(index=2, assigned_service_route=route)
    for vessel in (first_vessel, second_vessel):
        vessel.current_segment = route.segments[0]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("strategy must not be called when ENABLE_STRATEGY is False")

    monkeypatch.setattr(berth_idle_module, "ENABLE_STRATEGY", False)
    monkeypatch.setattr(
        UserStrategy, "select_vessel_for_berth", staticmethod(fail_if_called)
    )
    monkeypatch.setattr(
        DefaultStrategy, "select_vessel_for_berth", staticmethod(fail_if_called)
    )

    activity = BerthIdle(context)
    activity.d_loads_ready_finish.add(berth)
    activity.q_finish_signals.add(first_vessel)
    activity.q_finish_signals.add(second_vessel)
    activity.attempt_finish()

    assert berth.occupying_vessel is first_vessel
    assert first_vessel not in activity.q_finish_signals
    assert second_vessel in activity.q_finish_signals


def test_disabled_strategy_initial_booking_ignores_disruptions(monkeypatch):
    monkeypatch.setattr(origin_activity_module, "ENABLE_STRATEGY", False)
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "D")}
    context.ports.extend(ports.values())
    blocked_route = _add_route(
        context, "BLOCKED", [("A", "B", 1), ("B", "D", 1)]
    )
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=blocked_route.segments[0].associated_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )
    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("strategy must not be called when ENABLE_STRATEGY is False")

    monkeypatch.setattr(
        UserStrategy, "assign_associated_bookings", staticmethod(fail_if_called)
    )
    monkeypatch.setattr(
        DefaultStrategy, "assign_associated_bookings", staticmethod(fail_if_called)
    )
    activity = ShipmentWaitingForLoadingAtOriginPort(context)
    assigned = activity._assign_associated_bookings(shipment)

    assert assigned is True
    assert shipment.associated_bookings[0].service_route is blocked_route


def test_disabled_strategy_does_not_adjust_carried_shipment_bookings(monkeypatch):
    monkeypatch.setattr(vessel_queue_module, "ENABLE_STRATEGY", False)
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())
    route = _add_route(
        context, "ROUTE", [("A", "B", 1), ("B", "C", 1), ("C", "D", 1)]
    )
    berth = Berth(index=0, port=ports["C"])
    ports["C"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )
    shipment = Shipment(index=1, teu_size=1)
    booking = Booking(1, shipment, route, 1, 3)
    shipment.associated_bookings.append(booking)
    shipment.current_booking_index = 1
    vessel = Vessel(index=1, assigned_service_route=route)
    vessel.current_segment = route.segments[0]
    vessel.carried_shipments.append(shipment)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("strategy must not be called when ENABLE_STRATEGY is False")

    monkeypatch.setattr(
        UserStrategy,
        "adjust_bookings_before_cargo_handling",
        staticmethod(fail_if_called),
    )
    monkeypatch.setattr(
        DefaultStrategy,
        "adjust_bookings_before_cargo_handling",
        staticmethod(fail_if_called),
    )
    activity = VesselQueuingForBerth(context)
    activity._adjust_carried_shipments_before_cargo_handling(vessel)

    assert shipment.associated_bookings == [booking]
    assert booking.departure_segment_index == 1
    assert booking.arrival_segment_index == 3


def test_default_strategy_builds_one_alternative_route_from_existing_legs():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())
    source_route = _add_route(
        context, "SOURCE", [("A", "B", 1), ("B", "C", 1), ("C", "A", 1)]
    )
    context.initial_service_routes.append(source_route)
    detour_legs = [
        _add_existing_leg(context, ports["B"], ports["D"], 2),
        _add_existing_leg(context, ports["D"], ports["C"], 2),
    ]
    vessel = Vessel(index=1, assigned_service_route=source_route)
    context.vessels.append(vessel)
    source_route.deployed_vessels.append(vessel)
    congested_leg = source_route.segments[1].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )
    demand = Demand(origin_port=ports["B"], destination_port=ports["C"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)
    now = dt.datetime.min + dt.timedelta(days=1)

    DefaultStrategy.create_alternative_service_routes(context, now)

    alternative_routes = [
        route for route in context.service_routes
        if route.source_service_route is source_route
    ]
    assert len(alternative_routes) == 1
    alternative_route = alternative_routes[0]
    assert vessel.pending_assigned_service_route is alternative_route
    assert not alternative_route.deployed_vessels
    assert not DefaultStrategy.assign_associated_bookings(context, now, shipment)
    assert shipment.associated_bookings == []

    vessel.current_segment = source_route.segments[-1]
    vessel.current_segment.current_vessels.append(vessel)
    DefaultStrategy.create_alternative_service_routes(context, now, vessel)

    assert vessel.assigned_service_route is alternative_route
    assert DefaultStrategy.assign_associated_bookings(context, now, shipment)
    assert shipment.associated_bookings[0].service_route is alternative_route
    assert all(
        segment.associated_leg in context.legs
        for segment in alternative_route.segments
    )
    assert all(
        segment.associated_leg is not congested_leg
        for segment in alternative_route.segments
    )
    assert all(leg in [segment.associated_leg for segment in alternative_route.segments]
               for leg in detour_legs)

    DefaultStrategy.create_alternative_service_routes(context, now)
    assert len(context.service_routes) == 2


def test_default_strategy_alternative_route_avoids_closed_port():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())
    source_route = _add_route(
        context,
        "SOURCE",
        [("A", "B", 1), ("B", "C", 1), ("C", "D", 1), ("D", "A", 1)],
    )
    context.initial_service_routes.append(source_route)
    _add_existing_leg(context, ports["A"], ports["C"], 2)
    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    DefaultStrategy.create_alternative_service_routes(
        context, dt.datetime.min + dt.timedelta(days=1)
    )

    alternative_route = next(
        route for route in context.service_routes
        if route.source_service_route is source_route
    )
    assert all(
        segment.associated_leg.departure_port is not ports["B"]
        and segment.associated_leg.arrival_port is not ports["B"]
        for segment in alternative_route.segments
    )


def test_empty_vessel_switches_to_pending_route_at_start_port():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())
    source_route = _add_route(
        context, "SOURCE", [("A", "B", 1), ("B", "C", 1), ("C", "A", 1)]
    )
    context.initial_service_routes.append(source_route)
    _add_existing_leg(context, ports["B"], ports["D"], 2)
    _add_existing_leg(context, ports["D"], ports["C"], 2)
    vessel = Vessel(index=1, assigned_service_route=source_route)
    vessel.current_segment = source_route.segments[-1]
    vessel.current_segment.current_vessels.append(vessel)
    context.vessels.append(vessel)
    source_route.deployed_vessels.append(vessel)
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=source_route.segments[1].associated_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )
    now = dt.datetime.min + dt.timedelta(days=1)
    DefaultStrategy.create_alternative_service_routes(context, now)
    pending_route = vessel.pending_assigned_service_route

    carried_shipment = Shipment(index=99, teu_size=1)
    vessel.carried_shipments.append(carried_shipment)
    DefaultStrategy.create_alternative_service_routes(context, now, vessel)
    assert vessel.assigned_service_route is source_route
    assert vessel.pending_assigned_service_route is pending_route

    vessel.carried_shipments.remove(carried_shipment)
    DefaultStrategy.create_alternative_service_routes(context, now, vessel)

    assert vessel.assigned_service_route is pending_route
    assert vessel.pending_assigned_service_route is None
    assert vessel.current_segment is None
    assert vessel not in source_route.deployed_vessels
    assert vessel not in source_route.segments[-1].current_vessels
    assert vessel in pending_route.deployed_vessels


def test_empty_vessel_restores_to_source_route_after_disruption_ends():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())
    source_route = _add_route(
        context, "SOURCE", [("A", "B", 1), ("B", "C", 1), ("C", "A", 1)]
    )
    context.initial_service_routes.append(source_route)
    _add_existing_leg(context, ports["B"], ports["D"], 2)
    _add_existing_leg(context, ports["D"], ports["C"], 2)
    vessel = Vessel(index=1, assigned_service_route=source_route)
    vessel.current_segment = source_route.segments[-1]
    vessel.current_segment.current_vessels.append(vessel)
    context.vessels.append(vessel)
    source_route.deployed_vessels.append(vessel)
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=source_route.segments[1].associated_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )
    active_time = dt.datetime.min + dt.timedelta(days=1)
    DefaultStrategy.create_alternative_service_routes(context, active_time, vessel)
    alternative_route = vessel.assigned_service_route

    assert alternative_route.source_service_route is source_route
    assert vessel in alternative_route.deployed_vessels
    assert vessel not in source_route.deployed_vessels

    alternative_arrival_at_c = next(
        segment
        for segment in alternative_route.segments
        if segment.associated_leg.arrival_port is ports["C"]
    )
    vessel.current_segment = alternative_arrival_at_c
    alternative_arrival_at_c.current_vessels.append(vessel)

    ended_time = dt.datetime.min + dt.timedelta(days=3)
    DefaultStrategy.create_alternative_service_routes(context, ended_time, vessel)

    assert vessel.assigned_service_route is source_route
    assert vessel.pending_assigned_service_route is None
    assert vessel.current_segment is source_route.segments[1]
    assert vessel.get_next_segment() is source_route.segments[2]
    assert vessel not in alternative_route.deployed_vessels
    assert vessel in source_route.deployed_vessels


def test_vessel_arrival_strategy_reroutes_remaining_bookings_around_closed_port():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())

    current_route = _add_route(context, "CURRENT", [("D", "A", 1)])
    blocked_route = _add_route(
        context, "BLOCKED", [("A", "B", 1), ("B", "D", 1)]
    )
    alternate_route = _add_route(context, "ALTERNATE", [("A", "C", 2), ("C", "D", 2)])

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    current_booking = Booking(1, shipment, current_route, 1, 1)
    final_booking = Booking(2, shipment, blocked_route, 1, 2)
    shipment.associated_bookings.extend([current_booking, final_booking])
    current_route.associated_bookings.append(current_booking)
    blocked_route.associated_bookings.append(final_booking)
    shipment.current_booking_index = 1

    _adjust_after_current_booking_arrival(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert len(shipment.associated_bookings) == 3
    assert all(
        booking.service_route is alternate_route
        for booking in shipment.associated_bookings[1:]
    )
    assert current_booking in current_route.associated_bookings
    assert final_booking not in blocked_route.associated_bookings
    assert alternate_route.associated_bookings == shipment.associated_bookings[1:]
    assert shipment.current_booking_index == 1


def test_vessel_arrival_strategy_skips_unaffected_remaining_bookings():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())

    current_route = _add_route(context, "CURRENT", [("D", "A", 1)])
    direct_route = _add_route(context, "DIRECT", [("A", "D", 1)])
    _add_route(context, "ALTERNATE", [("A", "C", 2), ("C", "D", 2)])

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    current_booking = Booking(1, shipment, current_route, 1, 1)
    final_booking = Booking(2, shipment, direct_route, 1, 1)
    shipment.associated_bookings.extend([current_booking, final_booking])
    shipment.current_booking_index = 1

    _adjust_after_current_booking_arrival(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert shipment.associated_bookings == [current_booking, final_booking]
    assert shipment.current_booking_index == 1


def test_vessel_arrival_strategy_reroutes_remaining_bookings_around_congested_leg():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())

    current_route = _add_route(context, "CURRENT", [("D", "A", 1)])
    congested_route = _add_route(context, "CONGESTED", [("A", "B", 1), ("B", "D", 1)])
    alternate_route = _add_route(context, "ALTERNATE", [("A", "C", 2), ("C", "D", 2)])
    congested_leg = congested_route.segments[0].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    current_booking = Booking(1, shipment, current_route, 1, 1)
    final_booking = Booking(2, shipment, congested_route, 1, 2)
    shipment.associated_bookings.extend([current_booking, final_booking])
    shipment.current_booking_index = 1

    _adjust_after_current_booking_arrival(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert len(shipment.associated_bookings) == 3
    assert all(
        booking.service_route is alternate_route
        for booking in shipment.associated_bookings[1:]
    )
    assert shipment.current_booking_index == 1


def test_vessel_arrival_strategy_ignores_leg_outside_remaining_booking_range():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D", "E")}
    context.ports.extend(ports.values())

    current_route = _add_route(context, "CURRENT", [("E", "A", 1)])
    remaining_route = _add_route(
        context,
        "REMAINING",
        [("A", "B", 1), ("B", "C", 1), ("C", "D", 1), ("D", "A", 1)],
    )
    alternate_route = _add_route(context, "ALTERNATE", [("A", "E", 2), ("E", "D", 2)])
    congested_leg = remaining_route.segments[3].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    current_booking = Booking(1, shipment, current_route, 1, 1)
    final_booking = Booking(2, shipment, remaining_route, 1, 3)
    shipment.associated_bookings.extend([current_booking, final_booking])
    shipment.current_booking_index = 1

    _adjust_after_current_booking_arrival(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert shipment.associated_bookings == [current_booking, final_booking]
    assert all(
        booking.service_route is not alternate_route
        for booking in shipment.associated_bookings
    )
    assert shipment.current_booking_index == 1


def test_default_initial_assignment_avoids_closed_port_when_possible():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())

    blocked_route = _add_route(
        context, "BLOCKED", [("A", "B", 1), ("B", "D", 1)]
    )
    alternate_route = _add_route(context, "ALTERNATE", [("A", "C", 2), ("C", "D", 2)])

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)
    stale_booking = Booking(1, shipment, blocked_route, 1, 2)
    shipment.associated_bookings.append(stale_booking)
    blocked_route.associated_bookings.append(stale_booking)

    DefaultStrategy.assign_associated_bookings(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert shipment.associated_bookings
    assert all(
        booking.service_route is alternate_route
        for booking in shipment.associated_bookings
    )
    assert all(
        booking.service_route is not blocked_route
        for booking in shipment.associated_bookings
    )
    assert stale_booking not in blocked_route.associated_bookings
    assert alternate_route.associated_bookings == shipment.associated_bookings
    assert shipment.current_booking_index == 1


def test_default_initial_assignment_avoids_congested_leg_when_possible():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D")}
    context.ports.extend(ports.values())

    congested_route = _add_route(context, "CONGESTED", [("A", "B", 1), ("B", "D", 1)])
    alternate_route = _add_route(context, "ALTERNATE", [("A", "C", 2), ("C", "D", 2)])
    congested_leg = congested_route.segments[0].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )

    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)

    DefaultStrategy.assign_associated_bookings(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert shipment.associated_bookings
    assert all(
        booking.service_route is alternate_route
        for booking in shipment.associated_bookings
    )
    assert all(
        booking.service_route is not congested_route
        for booking in shipment.associated_bookings
    )
    assert shipment.current_booking_index == 1


def test_default_initial_assignment_fails_when_closed_port_is_unavoidable():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "D")}
    context.ports.extend(ports.values())

    _add_route(context, "BLOCKED", [("A", "B", 1), ("B", "D", 1)])

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)

    assigned = DefaultStrategy.assign_associated_bookings(
        context, dt.datetime.min + dt.timedelta(days=1), shipment
    )

    assert assigned is False
    assert shipment.associated_bookings == []
    assert shipment.current_booking_index is None


def test_origin_waiting_retries_after_avoid_port_reopens_when_no_path_exists():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "D")}
    context.ports.extend(ports.values())

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    context.disruption_plans.append(
        DisruptionPlan(
            target_berth=berth,
            start_offset_days=0,
            duration_days=2,
            close_berth=True,
        )
    )

    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)
    another_shipment = Shipment(index=2, teu_size=1, demand=demand)
    activity = ShipmentWaitingForLoadingAtOriginPort(context)
    activity.d_loads_ready_finish.add(shipment)
    activity.d_loads_ready_finish.add(another_shipment)
    scheduled_delays = []
    activity.schedule = lambda action, delay=None, tag=None: scheduled_delays.append(delay)

    activity.attempt_finish()

    assert shipment in activity.d_loads_ready_finish
    assert another_shipment in activity.d_loads_ready_finish
    assert shipment not in activity.f_loads_finished
    assert shipment.associated_bookings == []
    assert shipment.current_booking_index is None
    assert scheduled_delays == [dt.timedelta(days=2, seconds=1)]


def test_origin_waiting_retries_after_earliest_active_disruption_recovers():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "D")}
    context.ports.extend(ports.values())

    berth = Berth(index=0, port=ports["B"])
    ports["B"].berths.append(berth)
    congested_leg = Leg(ports["A"], ports["B"], 1)
    context.disruption_plans.extend(
        [
            DisruptionPlan(
                target_berth=berth,
                start_offset_days=0,
                duration_days=10,
                close_berth=True,
            ),
            DisruptionPlan(
                target_leg=congested_leg,
                start_offset_days=0,
                duration_days=2,
                multiplier=3,
            ),
        ]
    )

    demand = Demand(origin_port=ports["A"], destination_port=ports["D"])
    shipment = Shipment(index=1, teu_size=1, demand=demand)
    activity = ShipmentWaitingForLoadingAtOriginPort(context)
    activity.d_loads_ready_finish.add(shipment)
    scheduled_delays = []
    activity.schedule = lambda action, delay=None, tag=None: scheduled_delays.append(delay)

    activity.attempt_finish()

    assert shipment in activity.d_loads_ready_finish
    assert scheduled_delays == [dt.timedelta(days=2, seconds=1)]


def test_vessel_arrival_hook_shortens_current_booking_and_reroutes_carried_shipment():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B", "C", "D", "E")}
    context.ports.extend(ports.values())

    current_route = _add_route(
        context, "CURRENT", [("A", "B", 1), ("B", "C", 1), ("C", "D", 1)]
    )
    alternate_route = _add_route(context, "ALTERNATE", [("B", "E", 2), ("E", "D", 2)])
    congested_leg = current_route.segments[1].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    booking = Booking(1, shipment, current_route, 1, 3)
    shipment.associated_bookings.append(booking)
    current_route.associated_bookings.append(booking)
    shipment.current_booking_index = 1

    vessel = Vessel(index=1, assigned_service_route=current_route)
    vessel.current_segment = current_route.segments[0]
    vessel.carried_shipments.append(shipment)
    berth = Berth(index=0, port=ports["B"])
    berth.occupying_vessel = vessel

    activity = VesselQueuingForBerth(context)
    activity.d_loads_ready_finish.add(vessel)
    activity.q_finish_signals.add(berth)

    activity.attempt_finish()

    assert vessel in activity.f_loads_finished
    assert shipment.associated_bookings[0] is booking
    assert booking.arrival_segment_index == 1
    assert len(shipment.associated_bookings) == 3
    assert shipment.associated_bookings[1].service_route is alternate_route
    assert shipment.associated_bookings[1].departure_segment_index == 1
    assert shipment.associated_bookings[1].arrival_segment_index == 1
    assert shipment.associated_bookings[2].service_route is alternate_route
    assert shipment.associated_bookings[2].departure_segment_index == 2
    assert shipment.associated_bookings[2].arrival_segment_index == 2
    assert current_route.associated_bookings == [booking]
    assert alternate_route.associated_bookings == shipment.associated_bookings[1:]
    assert shipment.current_booking_index == 1


def test_vessel_arrival_hook_ignores_congested_leg_already_completed_in_current_booking():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("Shanghai", "Shenzhen", "Singapore", "Colombo")}
    context.ports.extend(ports.values())

    route = _add_route(
        context,
        "CURRENT",
        [
            ("Shanghai", "Shenzhen", 1),
            ("Shenzhen", "Singapore", 1),
            ("Singapore", "Colombo", 1),
            ("Colombo", "Shanghai", 1),
        ],
    )
    congested_leg = route.segments[1].associated_leg
    context.disruption_plans.append(
        DisruptionPlan(
            target_leg=congested_leg,
            start_offset_days=0,
            duration_days=2,
            multiplier=3,
        )
    )

    shipment = Shipment(index=1, teu_size=1)
    booking = Booking(1, shipment, route, 1, 3)
    shipment.associated_bookings.append(booking)
    shipment.current_booking_index = 1

    vessel = Vessel(index=1, assigned_service_route=route)
    vessel.current_segment = route.segments[2]
    vessel.carried_shipments.append(shipment)

    DefaultStrategy.adjust_bookings_before_cargo_handling(
        context, dt.datetime.min + dt.timedelta(days=1), vessel
    )

    assert shipment.associated_bookings == [booking]
    assert booking.departure_segment_index == 1
    assert booking.arrival_segment_index == 3
    assert shipment.current_booking_index == 1


def test_vessel_service_updates_current_storage_port_on_load_and_discharge():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B")}
    context.ports.extend(ports.values())
    route = _add_route(context, "ROUTE", [("A", "B", 1), ("B", "A", 1)])

    shipment = Shipment(
        index=1,
        teu_size=1,
        current_storage_port=ports["A"],
    )
    booking = Booking(1, shipment, route, 1, 1)
    shipment.associated_bookings.append(booking)
    shipment.current_booking_index = 1
    ports["A"].shipments_in_storage.append(shipment)

    vessel = Vessel(index=1, assigned_service_route=route)
    vessel.vessel_class = type("VesselClass", (), {"teu_capacity": 10})()
    activity = VesselBeingServed()
    activity.r_loads_requested_start.add(vessel)
    activity.p_start_signals.add(shipment)

    activity.attempt_start()

    assert shipment in vessel.carried_shipments
    assert shipment.current_storage_port is None
    assert shipment not in ports["A"].shipments_in_storage

    vessel.current_segment = route.segments[0]
    berth = Berth(index=0, port=ports["B"])
    vessel.current_berth = berth
    activity.d_loads_ready_finish.add(vessel)
    activity.q_finish_signals.add(berth)

    activity.attempt_finish()

    assert shipment not in vessel.carried_shipments
    assert shipment.current_storage_port is ports["B"]
    assert shipment in ports["B"].shipments_in_storage


def test_berth_assignment_keeps_queue_order_when_port_is_not_congested():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B")}
    context.ports.extend(ports.values())
    route = _add_route(context, "ROUTE", [("A", "B", 1), ("B", "A", 1)])

    berth = Berth(index=0, port=ports["B"])
    small_vessel = Vessel(index=1, assigned_service_route=route)
    small_vessel.current_segment = route.segments[0]
    small_vessel.vessel_class = type("VesselClass", (), {"teu_capacity": 100})()
    large_vessel = Vessel(index=2, assigned_service_route=route)
    large_vessel.current_segment = route.segments[0]
    large_vessel.vessel_class = type("VesselClass", (), {"teu_capacity": 300})()

    activity = BerthIdle(context)
    activity.d_loads_ready_finish.add(berth)
    activity.q_finish_signals.add(small_vessel)
    activity.q_finish_signals.add(large_vessel)

    activity.attempt_finish()

    assert berth.occupying_vessel is small_vessel
    assert small_vessel.current_berth is berth
    assert small_vessel not in activity.q_finish_signals
    assert large_vessel in activity.q_finish_signals


def test_port_response_strategy_runs_at_congestion_threshold():
    context = MaritimeDataContext()
    ports = {name: Port(name=name) for name in ("A", "B")}
    context.ports.extend(ports.values())
    route = _add_route(context, "ROUTE", [("A", "B", 1), ("B", "A", 1)])

    berth = Berth(index=0, port=ports["B"])
    vessels = []
    for index, capacity in enumerate((100, 200, 300), start=1):
        vessel = Vessel(index=index, assigned_service_route=route)
        vessel.current_segment = route.segments[0]
        vessel.vessel_class = type(
            "VesselClass",
            (),
            {"teu_capacity": capacity},
        )()
        vessels.append(vessel)

    activity = BerthIdle(context)
    activity.d_loads_ready_finish.add(berth)
    for vessel in vessels:
        activity.q_finish_signals.add(vessel)

    activity.attempt_finish()

    largest_vessel = vessels[-1]
    assert berth.occupying_vessel is largest_vessel
    assert largest_vessel.current_berth is berth
    assert largest_vessel not in activity.q_finish_signals


def test_port_response_strategy_hybrid_score_rewards_longer_waiting_time():
    context = MaritimeDataContext()
    port = Port(name="A")
    context.ports.append(port)
    small_vessel = Vessel(index=1)
    small_vessel.vessel_class = type("VesselClass", (), {"teu_capacity": 100})()
    large_vessel = Vessel(index=2)
    large_vessel.vessel_class = type("VesselClass", (), {"teu_capacity": 300})()
    now = dt.datetime.min + dt.timedelta(hours=10)

    selected = DefaultStrategy.select_vessel_for_berth(
        maritime_data_context=context,
        port=port,
        waiting_vessels=[small_vessel, large_vessel],
        available_berths=[],
        current_time=now,
        waiting_since_by_vessel={
            small_vessel: dt.datetime.min,
            large_vessel: now,
        },
    )

    assert selected is small_vessel


def _adjust_after_current_booking_arrival(context, now, shipment):
    current_booking = shipment.get_current_booking()
    current_segment = next(
        segment
        for segment in current_booking.service_route.segments
        if segment.sequence_index == current_booking.arrival_segment_index
    )
    vessel = Vessel(index=1, assigned_service_route=current_booking.service_route)
    vessel.current_segment = current_segment
    vessel.carried_shipments.append(shipment)
    DefaultStrategy.adjust_bookings_before_cargo_handling(context, now, vessel)


def _add_existing_leg(context, departure_port, arrival_port, distance):
    leg = Leg(departure_port, arrival_port, distance)
    context.legs.append(leg)
    departure_port.outgoing_legs.append(leg)
    arrival_port.incoming_legs.append(leg)
    return leg


def _add_route(context, route_id, legs):
    route = ServiceRoute(id=route_id, name=route_id)
    context.service_routes.append(route)
    for sequence_index, (departure_name, arrival_name, distance) in enumerate(legs, start=1):
        departure = next(port for port in context.ports if port.name == departure_name)
        arrival = next(port for port in context.ports if port.name == arrival_name)
        leg = Leg(departure, arrival, distance)
        segment = Segment(sequence_index, leg, route)
        leg.segments.append(segment)
        route.segments.append(segment)
        context.legs.append(leg)
    return route
