"""Baseline scenario plus port and sailing-leg disruptions."""

from config.simulation_config import WARM_UP_DAYS
from maritime_data_context import DisruptionPlan
from .baseline_stable_scenario import BaselineStableScenario

CONGESTED_LEGS = [
    # Formato: (puerto de salida, puerto de llegada, día de inicio, duración en días, multiplicador)
    ("Colombo", "New Jersey", 30.0, 90.0, 8.0),   
    ("Shanghai", "Shenzhen", 120.0, 75.0, 6.0),  
    ("Qingdao", "Shanghai", 200.0, 40.0, 6.0),   
    ("Singapore", "Colombo", 150.0, 50.0, 5.0),  # Dirección correcta (Singapore -> Colombo)
    ("Colombo", "Singapore", 220.0, 45.0, 5.0),  # Dirección inversa si aplica, o usa un tramo existente
]

CLOSED_PORTS = [
    # Formato: (nombre del puerto, día de inicio, duración en días)
    ("Piraeus", 250.0, 21.0),    # Puerto válido con cierre prolongado
    ("Tianjin", 310.0, 14.0),    # Cierre temporal en puerto norteasiático
    ("Shanghai", 180.0, 10.0),   # Puerto válido con gran volumen de carga
    ("Rotterdam", 280.0, 7.0),   # Puerto europeo clave de destino
]


def create_with_disruption(warm_up_days=WARM_UP_DAYS):
    """Create disruptions whose configured start days are measurement-relative."""
    context = BaselineStableScenario.create()

    for congested_leg in CONGESTED_LEGS:
        departure, arrival, start_day, duration, multiplier = congested_leg
        _add_congested_leg(
            context,
            departure,
            arrival,
            warm_up_days + start_day,
            duration,
            multiplier,
        )

    for closed_port in CLOSED_PORTS:
        port, start_day, duration = closed_port
        _add_closed_port(
            context,
            port,
            warm_up_days + start_day,
            duration,
        )

    return context


def _add_congested_leg(
    context,
    departure_port_name,
    arrival_port_name,
    start_offset_days,
    duration_days,
    multiplier,
):
    _validate_timing(start_offset_days, duration_days)
    if multiplier <= 1.0:
        raise ValueError("A congested-leg multiplier must be greater than 1.0.")

    legs = _require_legs(context, departure_port_name, arrival_port_name)

    for leg in legs:
        context.disruption_plans.append(
            DisruptionPlan(
                target_leg=leg,
                start_offset_days=start_offset_days,
                duration_days=duration_days,
                multiplier=multiplier,
            )
        )


def _add_closed_port(
    context,
    port_name,
    start_offset_days,
    duration_days,
):
    _validate_timing(start_offset_days, duration_days)
    port = _require_port(context, port_name)
    if not port.berths:
        raise ValueError(
            f"Port '{port.name}' does not have any berths to close."
        )

    for berth in port.berths:
        context.disruption_plans.append(
            DisruptionPlan(
                target_berth=berth,
                start_offset_days=start_offset_days,
                duration_days=duration_days,
                close_berth=True,
            )
        )


def _validate_timing(start_offset_days, duration_days):
    if start_offset_days < 0:
        raise ValueError("start_offset_days must be non-negative.")
    if duration_days <= 0:
        raise ValueError("duration_days must be positive.")


def _find_legs(context, departure_port_name, arrival_port_name):
    departure = departure_port_name.casefold()
    arrival = arrival_port_name.casefold()
    return [
        leg
        for leg in context.legs
        if leg.departure_port.name.casefold() == departure
        and leg.arrival_port.name.casefold() == arrival
    ]


def _require_legs(context, departure_port_name, arrival_port_name):
    legs = _find_legs(context, departure_port_name, arrival_port_name)
    if not legs:
        raise ValueError(
            f"Leg '{departure_port_name}' -> '{arrival_port_name}' "
            "was not found in the baseline scenario."
        )
    return legs


def _find_port(context, port_name):
    name = port_name.casefold()
    return next(
        (port for port in context.ports if port.name.casefold() == name),
        None,
    )


def _require_port(context, port_name):
    port = _find_port(context, port_name)
    if port is None:
        raise ValueError(f"Port '{port_name}' was not found in the baseline scenario.")
    return port
