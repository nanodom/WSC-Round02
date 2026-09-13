import pytest

from maritime_data_context import (
    Demand, Leg, MaritimeDataContext, Port, Segment, ServiceRoute, Shipment, Vessel,
)
from response_strategies.user_strategy import UserStrategy
from response_strategies.strategy_validation import (
    capture_alternative_route_strategy_state,
    validate_alternative_route_strategy_result,
)
from simulation_model.shipment_waiting_for_loading_at_origin_port import (
    ShipmentWaitingForLoadingAtOriginPort,
)


def test_validation_accepts_existing_leg_and_vessel_transfer():
    context, old_route, vessel, legs = _build_context()
    snapshot = capture_alternative_route_strategy_state(context)
    new_route = _add_route(context, "NEW", legs)
    old_route.deployed_vessels.remove(vessel)
    new_route.deployed_vessels.append(vessel)
    vessel.assigned_service_route = new_route

    validate_alternative_route_strategy_result(context, snapshot)


def test_validation_rejects_replaced_vessel_even_when_count_is_unchanged():
    context, _, _, _ = _build_context()
    snapshot = capture_alternative_route_strategy_state(context)
    context.vessels[:] = [Vessel(index=99)]

    with pytest.raises(ValueError, match="context.vessels must remain unchanged"):
        validate_alternative_route_strategy_result(context, snapshot)


def test_validation_rejects_new_leg_used_without_registering_it_in_context():
    context, _, _, _ = _build_context()
    snapshot = capture_alternative_route_strategy_state(context)
    port_a, port_b = context.ports
    _add_route(
        context,
        "NEW",
        [Leg(port_a, port_b, 1), Leg(port_b, port_a, 1)],
    )

    with pytest.raises(ValueError, match="must use only legs that existed"):
        validate_alternative_route_strategy_result(context, snapshot)


def test_origin_validates_user_changes_even_when_strategy_returns_none(monkeypatch):
    context, _, _, _ = _build_context()
    demand = Demand(origin_port=context.ports[0], destination_port=context.ports[1])
    shipment = Shipment(index=1, teu_size=1, demand=demand)

    def mutate_context_and_fall_back(context, now, vessel=None):
        context.vessels.append(Vessel(index=99))
        return None

    monkeypatch.setattr(
        UserStrategy,
        "create_alternative_service_routes",
        staticmethod(mutate_context_and_fall_back),
    )
    activity = ShipmentWaitingForLoadingAtOriginPort(context)

    with pytest.raises(ValueError, match="context.vessels must remain unchanged"):
        activity._assign_associated_bookings(shipment)


def _build_context():
    context = MaritimeDataContext()
    port_a, port_b = Port(name="A"), Port(name="B")
    context.ports.extend([port_a, port_b])
    legs = [Leg(port_a, port_b, 1), Leg(port_b, port_a, 1)]
    context.legs.extend(legs)
    route = _add_route(context, "ORIGINAL", legs)
    vessel = Vessel(index=1, assigned_service_route=route)
    context.vessels.append(vessel)
    route.deployed_vessels.append(vessel)
    return context, route, vessel, legs


def _add_route(context, route_id, legs):
    route = ServiceRoute(id=route_id, name=route_id)
    for index, leg in enumerate(legs, start=1):
        segment = Segment(index, leg, route)
        route.segments.append(segment)
        leg.segments.append(segment)
    context.service_routes.append(route)
    return route
