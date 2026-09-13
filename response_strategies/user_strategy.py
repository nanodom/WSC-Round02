"""Contestant strategy — WSC 2026 Simulation Challenge, Round 1 (merged).

Combines interventions from two contributors that started out independent
and now share signals across each other's hooks:

  * ``select_vessel_for_berth`` (JU, now network-aware) — convoy-
    coordination priority rule for congested-port berth assignment, now
    also weighted by how clear each vessel's next leg is (Section 4), using
    JP's own congestion check and Section 3's bypass-route cache.
  * ``create_alternative_service_routes`` (Section 3) — segment-skipping
    bypass routes for empty vessels facing a severely congested next leg.
  * ``adjust_bookings_before_cargo_handling`` (JP) — "wait out the
    disruption window when waiting is arithmetically better" holding rule.
  * ``assign_associated_bookings`` (Section 5) — same shortest-path search
    as DefaultStrategy, reused directly, but augmented so it never offers a
    suspended (zero-vessel) route and does offer Section 3's bypass routes.

Every decision point that isn't explicitly implemented still returns
``None`` and falls back to ``DefaultStrategy``:

  select_vessel_for_berth              -> JU's convoy logic + Section 4 network-awareness
  create_alternative_service_routes    -> Section 3 segment-skipping bypass, else None
  assign_associated_bookings           -> Section 5 augmented shortest-path search
  adjust_bookings_before_cargo_handling -> JP's hold/split logic

See each section's docstring below for the rationale and how the pieces
hand signals to one another.

Resilience notes (added for catastrophic / overlapping-disruption scenarios)
------------------------------------------------------------------------
1. Every public entry point is wrapped with ``@_safe_strategy`` so that an
   unexpected exception in any of the four hooks never aborts the whole
   simulation run. It is logged and the method degrades to the same
   "no opinion, use DefaultStrategy" fallback value the un-decorated
   contract already used for its ordinary "nothing to do" case.
2. Every stateful intervention (holding a booking, building a bypass
   route) is now bounded by an explicit budget (see the ``MAX_*``
   constants below). Once a budget is exhausted the strategy simply stops
   intervening further and lets DefaultStrategy take over for the
   remainder — it never grows state without bound across a long run with
   many overlapping disruptions.
"""

import datetime as dt
import logging

from maritime_data_context import Booking, Segment, ServiceRoute
from response_strategies import default_strategy as _default_strategy
from simulation_model.disruption_status import is_disruption_active
from simulation_model.ordered_set import OrderedSet

_logger = logging.getLogger(__name__)


# =============================================================================
# Resilience helper: exception safety decorator
# =============================================================================
#
# Every one of the four strategy hooks below has a well-defined "I have no
# opinion, fall back to DefaultStrategy" contract already (None for three of
# them, False for assign_associated_bookings). This decorator makes sure an
# *unexpected* exception is treated exactly the same way as that existing
# "no opinion" contract, instead of propagating up and aborting -- and
# therefore disqualifying -- the whole simulation run. The exception is
# logged (with traceback) so a real bug is still visible during development,
# but it can never take down the run.

def _safe_strategy(default_return=None):
    """Decorator: catch any exception from the wrapped strategy hook, log
    it, and return ``default_return`` instead of letting it propagate.

    ``default_return`` must match the hook's own "no opinion" contract:
    ``None`` for select_vessel_for_berth / create_alternative_service_routes
    / adjust_bookings_before_cargo_handling, and ``False`` for
    assign_associated_bookings (which means "no path found, keep the
    shipment waiting and retry later" rather than "delegate entirely").
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                _logger.exception(
                    "UserStrategy.%s raised an exception; falling back to "
                    "default_return=%r for this call.",
                    func.__name__,
                    default_return,
                )
                return default_return

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# =============================================================================
# Section 1: JU — Convoy-coordination helpers for select_vessel_for_berth
# =============================================================================
#
# select_vessel_for_berth can only return ONE vessel per call (one berth
# frees up at a time) — there's no hook to reserve several berths for a
# convoy at once. What we approximate instead: convoys are recomputed from
# the live `waiting_vessels` snapshot on every call, and that snapshot only
# shrinks as vessels get served, so the same convoy keeps winning priority
# call after call until it's exhausted. That's what produces the
# "coordinated single window" effect without touching berth_idle.py.

_ROUTE_SIMILARITY_WEIGHT = 0.6
_SHARED_PORT_WEIGHT = 0.4
_CONVOY_MEMBERSHIP_THRESHOLD = 0.5      # composite index needed to join a convoy
_SMALL_CONVOY_MAX_SIZE = 3              # cap under "medium" congestion
_SEVERE_CONGESTION_RATIO = 6.0          # waiting_vessels / available_berths


def _route_ports(vessel):
    """Set of port names touched anywhere along the vessel's assigned route."""
    route = vessel.assigned_service_route
    if route is None:
        return frozenset()
    ports = set()
    for segment in route.segments:
        leg = segment.associated_leg
        if leg is None:
            continue
        ports.add(leg.departure_port.name)
        ports.add(leg.arrival_port.name)
    return frozenset(ports)


def _route_similarity(ports_a, ports_b):
    """Jaccard similarity between two routes' port sets."""
    union = ports_a | ports_b
    if not union:
        return 0.0
    return len(ports_a & ports_b) / len(union)


def _shared_port_pct(ports_a, ports_b):
    """Share of the smaller route's ports also present on the other route."""
    smaller = min(len(ports_a), len(ports_b))
    if smaller == 0:
        return 0.0
    return len(ports_a & ports_b) / smaller


def _compatibility_index(ports_a, ports_b):
    """Composite convoy-membership score in [0, 1]."""
    return (
        _ROUTE_SIMILARITY_WEIGHT * _route_similarity(ports_a, ports_b)
        + _SHARED_PORT_WEIGHT * _shared_port_pct(ports_a, ports_b)
    )


def _congestion_mode(waiting_vessels, available_berths):
    """"small" -> tightly-matched, bounded convoys.
    "fleet"  -> severe congestion, treat all waiting vessels as one convoy.
    """
    berth_count = max(1, len(available_berths))
    ratio = len(waiting_vessels) / berth_count
    return "fleet" if ratio >= _SEVERE_CONGESTION_RATIO else "small"


def _build_convoys(ordered_vessels, route_ports_by_vessel, max_size):
    """Greedily group vessels (oldest-waiting first) into convoys.

    Each vessel joins the best-matching existing convoy (compared against
    that convoy's anchor, its first/oldest member) if the compatibility
    index clears the threshold and there's room; otherwise it starts a
    new convoy.
    """
    convoys = []
    for vessel in ordered_vessels:
        vessel_ports = route_ports_by_vessel[vessel]
        best_convoy, best_score = None, 0.0
        for convoy in convoys:
            if max_size is not None and len(convoy) >= max_size:
                continue
            anchor_ports = route_ports_by_vessel[convoy[0]]
            score = _compatibility_index(vessel_ports, anchor_ports)
            if score >= _CONVOY_MEMBERSHIP_THRESHOLD and score > best_score:
                best_convoy, best_score = convoy, score
        if best_convoy is not None:
            best_convoy.append(vessel)
        else:
            convoys.append([vessel])
    return convoys


# =============================================================================
# Section 2: JP — Hold registry & tuning for adjust_bookings_before_cargo_handling
# =============================================================================
#
# The gap being exploited: DefaultStrategy.adjust_bookings_before_cargo_handling
# already reroutes cargo around congested legs, but only when an alternative
# path exists. When no reroute exists, the remaining option is to WAIT for the
# disruption window to expire — that is what this section adds, on top of
# (never instead of) DefaultStrategy's own rerouting.
#
# Waiting is only correct when the induced delay from waiting exceeds the
# time saved vs. sailing now under the multiplier. The decision is made per
# leg, per shipment, at the moment of departure, comparing:
#
#     sail now = distance / speed * multiplier
#     wait     = time until the window closes + distance / speed
#
# Only when waiting is strictly cheaper (by a margin) is the shipment held.

# Only hold cargo when waiting saves at least this many days. Keeps the policy
# away from marginal calls where the arithmetic is inside model noise.
MINIMUM_SAVING_DAYS = 5.0

# A hold is never allowed to outlive this, no matter what the plan says.
# Pure insurance against stranding cargo.
MAXIMUM_HOLD_DAYS = 45.0

# departure_segment_index value that matches no real segment (real ones are >= 1).
PARKED_SENTINEL = -9999

DEFAULT_SAILING_SPEED_KNOTS = 20.0

# --- Resilience: aggregate circuit-breaker budgets (point 5) ----------------
#
# The per-booking MAXIMUM_HOLD_DAYS guard above bounds how long any single
# hold can last, but says nothing about how MANY shipments can be held at
# once. With multiple overlapping disruptions across a full year, a very
# broad congestion event could otherwise cause an unbounded number of
# shipments to be parked simultaneously. These caps bound that aggregate
# footprint; once a cap is hit, the strategy simply declines to hold any
# further shipments and lets DefaultStrategy's own rerouting logic (or
# eventual retry once the window closes) handle the rest.

# Maximum number of bookings held (PARKED_SENTINEL'd) at the same time,
# across all ports combined.
MAX_HELD_BOOKINGS_TOTAL = 500

# Maximum number of bookings held at the same time for any single port, so
# one severely congested port cannot consume the entire global budget and
# starve holding decisions everywhere else.
MAX_HELD_BOOKINGS_PER_PORT = 100


# booking -> (original_departure_segment_index, release_time)
_HELD_BOOKINGS = {}

# Last simulation clock seen, used to detect a fresh run in the same process.
_LAST_SEEN_TIME = None


def _reset_if_new_run(now) -> None:
    """Drop state left over from a previous simulation in the same process.

    The evaluation harness is not ours, and it may well run several scenarios or
    seeds inside one Python process. Each run restarts the clock near
    datetime.min, so holds from a previous run would never reach their release
    time and would pile up forever against dead Booking objects.
    """
    global _LAST_SEEN_TIME
    if _LAST_SEEN_TIME is not None and now < _LAST_SEEN_TIME:
        _HELD_BOOKINGS.clear()
    _LAST_SEEN_TIME = now


# --- release -----------------------------------------------------------------

def _release_due_holds(now) -> None:
    """Restore every hold whose release time has arrived.

    Runs at the top of every call, before anything else, and depends only on
    the recorded absolute time.
    """
    if not _HELD_BOOKINGS:
        return

    due = [
        booking
        for booking, (_, release_time) in _HELD_BOOKINGS.items()
        if now >= release_time
    ]
    for booking in due:
        original_index, _ = _HELD_BOOKINGS.pop(booking)
        if booking.departure_segment_index == PARKED_SENTINEL:
            booking.departure_segment_index = original_index


# --- discharge cargo riding through -------------------------------------------

def _discharge_carried_cargo_that_should_wait(context, vessel, now) -> None:
    """Land cargo that would otherwise ride the disrupted leg while onboard.

    Reaches cargo that is in transit (not just cargo sitting in port storage):
    e.g. a Shanghai -> Los Angeles box booked through Kaohsiung onboard, never
    entering shipments_in_storage there.

    Splitting its booking in two -- current booking ends here, a continuation
    covers the rest -- makes it discharge at this port. From there
    _hold_departures_that_should_wait keeps it until the window closes, and the
    continuation booking puts it on a later vessel.

    Adding a continuation also makes is_at_last_booking() False, which routes
    the shipment to the transshipment-waiting activity instead of terminating it.
    """
    route = getattr(vessel, "assigned_service_route", None)
    current_segment = getattr(vessel, "current_segment", None)
    if route is None or current_segment is None:
        return

    try:
        next_segment = vessel.get_next_segment()
    except (ValueError, AttributeError):
        return
    if next_segment is None:
        return

    # Only the straightforward forward step. Wrap-around at the end of the
    # rotation would require renumbering across the cycle boundary; not worth
    # the risk for the gain.
    if next_segment.sequence_index != current_segment.sequence_index + 1:
        return

    leg = getattr(next_segment, "associated_leg", None)
    if leg is None:
        return

    for shipment in list(getattr(vessel, "carried_shipments", []) or []):
        try:
            booking = shipment.get_current_booking()
        except (ValueError, AttributeError):
            continue

        if booking.service_route is not route:
            continue
        if booking.departure_segment_index == PARKED_SENTINEL:
            continue
        # Already getting off here — nothing to do.
        if booking.arrival_segment_index <= current_segment.sequence_index:
            continue
        if _release_time_if_waiting_is_better(context, leg, booking, now) is None:
            continue

        _split_booking_at_current_port(shipment, booking, current_segment, next_segment)


def _split_booking_at_current_port(shipment, booking, current_segment, next_segment) -> None:
    """Cut `booking` in two at the current port, keeping the chain consistent."""
    final_arrival_index = booking.arrival_segment_index

    # Current booking now ends here -> the shipment discharges at this port.
    booking.arrival_segment_index = current_segment.sequence_index

    # Make room for the continuation. Done before appending it, so the
    # continuation itself is not shifted.
    for other in shipment.associated_bookings:
        if other.sequence_index > booking.sequence_index:
            other.sequence_index += 1

    continuation = Booking(
        sequence_index=booking.sequence_index + 1,
        shipment=shipment,
        service_route=booking.service_route,
        departure_segment_index=next_segment.sequence_index,
        arrival_segment_index=final_arrival_index,
    )
    shipment.associated_bookings.append(continuation)
    shipment.associated_bookings.sort(key=lambda item: item.sequence_index)

    route_bookings = getattr(continuation.service_route, "associated_bookings", None)
    if route_bookings is not None:
        route_bookings.append(continuation)


# --- hold ----------------------------------------------------------------------

def _booking_departure_port(booking):
    """Departure port of `booking`'s current departure segment, or None.

    Used only for the per-port budget accounting below; failures here
    should never block holding logic, so any lookup miss just returns None
    (meaning "don't count against a specific port's budget").
    """
    segment = _find_segment(booking.service_route, booking.departure_segment_index)
    if segment is None:
        return None
    leg = getattr(segment, "associated_leg", None)
    return getattr(leg, "departure_port", None) if leg is not None else None


def _held_bookings_count_for_port(port) -> int:
    return sum(
        1
        for held_booking in _HELD_BOOKINGS
        if _booking_departure_port(held_booking) is port
    )


def _hold_departures_that_should_wait(context, port, now) -> None:
    """Hold stored shipments at `port` whose next leg is worth waiting out.

    Bounded by MAX_HELD_BOOKINGS_TOTAL / MAX_HELD_BOOKINGS_PER_PORT so a
    single very broad disruption (or many overlapping ones across the year)
    cannot grow the held-booking registry without limit. Once a budget is
    hit, the strategy stops holding further shipments at this port for the
    rest of this call — DefaultStrategy's own rerouting (called separately)
    still gets a chance to act on them.
    """
    if len(_HELD_BOOKINGS) >= MAX_HELD_BOOKINGS_TOTAL:
        return

    port_held_count = _held_bookings_count_for_port(port)
    if port_held_count >= MAX_HELD_BOOKINGS_PER_PORT:
        return

    for shipment in list(getattr(port, "shipments_in_storage", []) or []):
        if len(_HELD_BOOKINGS) >= MAX_HELD_BOOKINGS_TOTAL:
            break
        if port_held_count >= MAX_HELD_BOOKINGS_PER_PORT:
            break

        try:
            booking = shipment.get_current_booking()
        except (ValueError, AttributeError):
            continue

        if booking in _HELD_BOOKINGS:
            continue
        if booking.departure_segment_index == PARKED_SENTINEL:
            continue

        segment = _find_segment(booking.service_route, booking.departure_segment_index)
        if segment is None:
            continue

        leg = getattr(segment, "associated_leg", None)
        if leg is None or leg.departure_port is not port:
            continue

        release_time = _release_time_if_waiting_is_better(context, leg, booking, now)
        if release_time is None:
            continue

        _HELD_BOOKINGS[booking] = (booking.departure_segment_index, release_time)
        booking.departure_segment_index = PARKED_SENTINEL
        port_held_count += 1


def _release_time_if_waiting_is_better(context, leg, booking, now):
    """Return when to release, or None if sailing now is the better option."""
    plan_end = _active_congestion_end(context, leg, now)
    if plan_end is None:
        return None

    multiplier = getattr(leg, "sailing_time_multiplier", 1.0) or 1.0
    if multiplier <= 1.0:
        return None

    speed = _route_sailing_speed(booking.service_route)
    distance = getattr(leg, "sailing_distance", 0.0) or 0.0
    if speed <= 0 or distance <= 0:
        return None

    normal_days = distance / speed / 24.0
    sail_now_days = normal_days * multiplier
    wait_days = max(0.0, (plan_end - now).total_seconds() / 86400.0)

    if wait_days > MAXIMUM_HOLD_DAYS:
        return None
    if wait_days + normal_days + MINIMUM_SAVING_DAYS >= sail_now_days:
        return None

    return plan_end


def _active_congestion_end(context, leg, now):
    """Latest end time among active congested-leg plans targeting `leg`."""
    latest = None
    for plan in getattr(context, "disruption_plans", []) or []:
        if getattr(plan, "target_leg", None) is not leg:
            continue
        if getattr(plan, "multiplier", 1.0) <= 1.0:
            continue
        start_offset = getattr(plan, "start_offset_days", None)
        duration = getattr(plan, "duration_days", None)
        if start_offset is None or duration is None:
            continue

        start = dt.datetime.min + dt.timedelta(days=start_offset)
        end = start + dt.timedelta(days=duration)
        if start <= now < end and (latest is None or end > latest):
            latest = end

    return latest


# --- small helpers ---------------------------------------------------------

def _get_vessel_current_port(vessel):
    """Port the vessel has just arrived at, or None."""
    segment = getattr(vessel, "current_segment", None)
    if segment is None:
        return None
    leg = getattr(segment, "associated_leg", None)
    return getattr(leg, "arrival_port", None) if leg is not None else None


def _find_segment(service_route, sequence_index):
    if service_route is None:
        return None
    for segment in getattr(service_route, "segments", []) or []:
        if segment.sequence_index == sequence_index:
            return segment
    return None


def _route_sailing_speed(service_route) -> float:
    """Sailing speed of the vessels on this route, in knots."""
    for vessel in getattr(service_route, "deployed_vessels", []) or []:
        vessel_class = getattr(vessel, "vessel_class", None)
        speed = getattr(vessel_class, "sailing_speed", None)
        if speed:
            return float(speed)
    return DEFAULT_SAILING_SPEED_KNOTS


# =============================================================================
# Section 3: Segment-Skipping bypass for create_alternative_service_routes
# =============================================================================
#
# Idea: instead of rebuilding a vessel's whole route the way DefaultStrategy
# does network-wide (for every vessel on an affected route, whenever
# is_disruption_active() is True), react to the *specific* congested leg a
# given vessel is about to sail into and, only for that one vessel, clone a
# one-off ServiceRoute that skips the single bottleneck port -- connecting
# the port just before it directly to the port just after it -- using only a
# leg that already exists in context.legs (never a new one). If no such
# direct leg exists, or skipping would leave the bottleneck port with no
# other service, the method backs off and returns None so DefaultStrategy's
# broader, multi-hop rerouting can take over safely.
#
# Because only ONE vessel is ever moved off the original route object, and
# the original route/segments are left completely untouched, every other
# vessel (and every future booking against that route) keeps using it
# exactly as before -- the bottleneck port is never actually isolated
# network-wide by this bypass.
#
# Safety scope -- empty vessels only: the switch is applied only to a vessel
# with no carried_shipments. Existing Bookings reference a fixed
# ``booking.service_route``, and both loading (attempt_start) and
# discharging (Vessel._get_shipments_at_segment) match against
# ``vessel.assigned_service_route``. Reassigning a vessel that is still
# carrying cargo booked against the original route would silently break that
# matching for those shipments. DefaultStrategy's own route-switch helpers
# (_try_switch_empty_vessel_to_pending_route / _to_source_route) apply this
# exact same empty-vessel-only restriction, so this mirrors an
# already-established safety pattern in the codebase rather than inventing a
# new one.
#
# Pairing with JP's holding logic: JP already knows, leg by leg, whether the
# active multiplier makes waiting worthwhile (``_active_congestion_end``,
# Section 2). We reuse that exact helper here, so a vessel is only bypassed
# around a leg that JP would also treat as "genuinely disrupted" -- the
# cargo JP holds at berth and the vessels this strategy reroutes are
# reacting to the identical signal, instead of two independently-tuned
# congestion checks disagreeing with each other.
#
# Pairing with JU's convoy scoring: ``_route_ports`` (Section 1) is
# evaluated against ``vessel.assigned_service_route`` at call time. A vessel
# bypassing a single port keeps every other port of its rotation, so its
# port-set stays almost identical to the rest of the fleet on the source
# route -- it is not kicked out of its convoy by this temporary reroute.
#
# Resilience: bypass creation is now also bounded (see MAX_ACTIVE_BYPASSES_
# TOTAL / MAX_BYPASS_ROUTES_PER_SOURCE_ROUTE below), so a year with many
# overlapping disruptions across many routes cannot make the number of live
# bypass ServiceRoutes -- and therefore the candidate-edge graph searched on
# every booking assignment -- grow without bound.

# --- Resilience: aggregate circuit-breaker budgets (point 5) ----------------

# Maximum number of vessels simultaneously diverted onto a bypass route.
MAX_ACTIVE_BYPASSES_TOTAL = 50

# Maximum number of distinct bypass routes ever created per source route,
# so that a route hit by disruptions at many different ports over the year
# doesn't accumulate an unbounded number of one-off ServiceRoute clones.
MAX_BYPASS_ROUTES_PER_SOURCE_ROUTE = 5

# Cache of built bypass routes, keyed by (source route id, skipped port
# name), so repeated congestion on the same route/port reuses one
# ServiceRoute instead of creating a new object on every call.
_BYPASS_ROUTES = {}

# vessel -> (original_route, bypass_route, skipped_leg)
_ACTIVE_BYPASS_BY_VESSEL = {}


def _peek_next_segment(vessel):
    """``vessel.get_next_segment()``, tolerating vessels with no route/segments yet."""
    try:
        return vessel.get_next_segment()
    except (ValueError, AttributeError):
        return None


def _port_has_other_service(port, route):
    """Whether any leg touching `port` is also served by a route other than `route`."""
    legs = list(getattr(port, "incoming_legs", []) or []) + list(getattr(port, "outgoing_legs", []) or [])
    for leg in legs:
        for segment in leg.segments:
            if segment.associated_service_route is not route:
                return True
    return False


def _safe_to_skip_port(route, skip_port):
    """Refuse to bypass a port that would otherwise be left with no service.

    Only this one vessel ever leaves `route` for the bypass route, so the
    port stays served as long as either another vessel remains on `route`
    (which keeps visiting it) or some other route also touches it.
    """
    return len(route.deployed_vessels) > 1 or _port_has_other_service(skip_port, route)


def _find_direct_leg(context, departure_port, arrival_port):
    """An existing Leg departure_port -> arrival_port, or None."""
    for leg in context.legs:
        if leg.departure_port is departure_port and leg.arrival_port is arrival_port:
            return leg
    return None


def _next_bypass_route_id(context, source_route_id, skip_port_name):
    existing_ids = {r.id.casefold() for r in context.service_routes}
    index = 1
    while True:
        candidate = f"{source_route_id}-BYPASS-{skip_port_name}-{index}"
        if candidate.casefold() not in existing_ids:
            return candidate
        index += 1


def _bypass_routes_count_for_source(source_route_id) -> int:
    return sum(
        1
        for (route_id, _skip_port_name) in _BYPASS_ROUTES
        if route_id == source_route_id
    )


def _build_bypass_route(context, route, next_segment, segment_after, bypass_leg, skip_port):
    """Clone `route`'s rotation with [next_segment, segment_after] replaced by one bypass_leg."""
    segments = sorted(route.segments, key=lambda s: s.sequence_index)
    n = len(segments)
    after_idx = segments.index(segment_after)

    kept_legs = []
    start = (after_idx + 1) % n
    for offset in range(n - 2):
        kept_legs.append(segments[(start + offset) % n].associated_leg)

    rotation_legs = [bypass_leg] + kept_legs

    new_route = ServiceRoute(
        id=_next_bypass_route_id(context, route.id, skip_port.name),
        name=f"{route.name} Bypass ({skip_port.name})",
        start_day_of_week=route.start_day_of_week,
    )
    new_route.source_service_route = route

    for sequence_index, leg in enumerate(rotation_legs, start=1):
        segment = Segment(sequence_index, leg, new_route)
        new_route.segments.append(segment)
        leg.segments.append(segment)
        context.partial_service_routes.append(segment)

    context.service_routes.append(new_route)
    return new_route


def _get_or_build_bypass_route(context, route, next_segment):
    """Return a cached/new bypass ServiceRoute skipping next_segment's port, or None
    if skipping is unsafe, over budget, or no direct existing leg makes it possible.
    """
    segments = sorted(route.segments, key=lambda s: s.sequence_index)
    if len(segments) < 3:
        return None  # nothing left to form a cycle around after skipping one port

    idx = segments.index(next_segment)
    after_idx = (idx + 1) % len(segments)
    segment_after = segments[after_idx]

    before_port = next_segment.associated_leg.departure_port
    skip_port = next_segment.associated_leg.arrival_port
    after_port = segment_after.associated_leg.arrival_port

    if skip_port is before_port or skip_port is after_port:
        return None  # degenerate rotation, nothing sensible to skip

    if not _safe_to_skip_port(route, skip_port):
        return None  # would isolate a destination -> fall back to DefaultStrategy

    cache_key = (route.id, skip_port.name.casefold())
    cached = _BYPASS_ROUTES.get(cache_key)
    if cached is not None and cached in context.service_routes:
        return cached

    # Resilience: only pay the budget check when we'd actually create a NEW
    # bypass route -- an already-cached one is free to keep reusing.
    if _bypass_routes_count_for_source(route.id) >= MAX_BYPASS_ROUTES_PER_SOURCE_ROUTE:
        return None  # budget exhausted for this source route -> fall back safely

    bypass_leg = _find_direct_leg(context, before_port, after_port)
    if bypass_leg is None:
        return None  # no viable alternative -> safe fallback to DefaultStrategy

    new_route = _build_bypass_route(context, route, next_segment, segment_after, bypass_leg, skip_port)
    _BYPASS_ROUTES[cache_key] = new_route
    return new_route


def _detach_vessel_from_route(vessel, from_route):
    current_segment = vessel.current_segment
    if current_segment is not None:
        while vessel in current_segment.current_vessels:
            current_segment.current_vessels.remove(vessel)
    while vessel in from_route.deployed_vessels:
        from_route.deployed_vessels.remove(vessel)


def _attach_vessel_to_route(vessel, to_route, segment):
    if vessel not in to_route.deployed_vessels:
        to_route.deployed_vessels.append(vessel)
    vessel.assigned_service_route = to_route
    vessel.pending_assigned_service_route = None
    vessel.current_segment = segment
    if vessel not in segment.current_vessels:
        segment.current_vessels.append(vessel)


def _find_reentry_segment(route, port):
    for segment in sorted(route.segments, key=lambda s: s.sequence_index):
        if segment.associated_leg.arrival_port is port:
            return segment
    return None


def _restore_vessels_whose_bypass_cleared(context, now):
    """Switch vessels back onto their original route once the bypassed leg's
    congestion has ended and the vessel is empty and back at the bypass's
    starting port (i.e. about to loop into the bypass leg again)."""
    for vessel, (original_route, bypass_route, leg) in list(_ACTIVE_BYPASS_BY_VESSEL.items()):
        if vessel.assigned_service_route is not bypass_route:
            _ACTIVE_BYPASS_BY_VESSEL.pop(vessel, None)
            continue
        if vessel.carried_shipments:
            continue  # wait until it's empty before touching its route again
        if _active_congestion_end(context, leg, now) is not None:
            continue  # still congested, keep bypassing

        current_port = None
        if vessel.current_segment is not None:
            current_port = vessel.current_segment.associated_leg.arrival_port
        reentry_segment = _find_reentry_segment(original_route, current_port) if current_port else None
        if reentry_segment is None:
            continue  # not currently at a port shared with the original route

        _detach_vessel_from_route(vessel, bypass_route)
        _attach_vessel_to_route(vessel, original_route, reentry_segment)
        _ACTIVE_BYPASS_BY_VESSEL.pop(vessel, None)


# =============================================================================
# Section 4: JU + JP + Section 3 — Next-leg awareness for select_vessel_for_berth
# =============================================================================
#
# select_vessel_for_berth only decided *who* to serve based on how long a
# vessel had waited and which convoy it belonged to. Neither signal says
# anything about what happens right after the berth is granted: a vessel
# sent out into a severely congested next leg just relocates the jam from
# this berth to the next bottleneck's queue or anchorage. This section adds
# that missing signal.
#
# The congestion check is the exact same one JP already trusts for the
# hold-vs-sail decision (_active_congestion_end, Section 2) -- so berthing
# priority and JP's holding logic are reading the identical trigger, instead
# of two independently-tuned congestion checks disagreeing with each other.
# A vessel with a *known* Section-3 bypass route around its bottleneck
# (looked up, never built, here -- building belongs to
# create_alternative_service_routes) is treated as better off than a truly
# blocked one, but not quite as good as a vessel with a genuinely open next
# leg.
#
# This does NOT change how convoys are grouped (still pure route similarity,
# Section 1) -- it changes which convoy is judged worth prioritizing, and
# which vessel within it goes first, by weighting each vessel's waiting time
# with its next-leg status. The weighting is multiplicative, not a hard
# veto: a vessel that has waited long enough still eventually outranks a
# fresher, "clearer" one, so a persistently blocked vessel is deprioritized
# rather than starved outright.

_NEXT_LEG_PRIORITY_WEIGHT = {
    "open": 1.20,        # next leg is clear -> evacuate the berth promptly
    "bypassable": 1.00,  # next leg is congested but a known bypass exists -> neutral
    "blocked": 0.55,     # next leg is severely congested, no bypass known -> hold back
}


def _vessel_next_leg_status(context, vessel, now):
    """"open" / "bypassable" / "blocked" for the leg the vessel will sail
    once it leaves this berth (``vessel.get_next_segment()`` from a vessel
    still waiting for berth resolves to exactly that leg -- see docstring
    of select_vessel_for_berth for why)."""
    next_segment = _peek_next_segment(vessel)
    if next_segment is None:
        return "open"  # no route/segment info yet -> don't penalize
    leg = next_segment.associated_leg
    if leg is None:
        return "open"
    if _active_congestion_end(context, leg, now) is None:
        return "open"  # not under an active, high-multiplier disruption

    route = vessel.assigned_service_route
    if route is not None:
        cache_key = (route.id, leg.arrival_port.name.casefold())
        if cache_key in _BYPASS_ROUTES:
            return "bypassable"  # a direct bypass around this exact bottleneck is known
    return "blocked"


# =============================================================================
# Section 5: Proactive rerouting for assign_associated_bookings
# =============================================================================
#
# assign_associated_bookings only runs at the shipment's ORIGIN port (called
# from shipment_waiting_for_loading_at_origin_port.attempt_finish, and
# re-called on retry once a disruption's window closes — see
# ``_schedule_booking_assignment_retry`` there). It builds the shipment's
# *entire* booking chain to destination in one shot via a Dijkstra-style
# shortest path over candidate edges, exactly like
# DefaultStrategy.assign_associated_bookings does. Rather than reimplement
# that graph search, we reuse DefaultStrategy's own private helpers
# (``_default_strategy``, imported above) for congestion-aware candidate
# edges and path-finding — this only relies on ``response_strategies``'s
# ``__init__.py`` importing ``default_strategy`` before ``user_strategy``
# (it already does), so DefaultStrategy's module is fully loaded by the time
# this module needs it.
#
# Two gaps we close on top of DefaultStrategy's own search:
#
#   1. Suspended original routes. DefaultStrategy only checks
#      ``deployed_vessels`` for routes with a ``source_service_route`` (i.e.
#      its own alternates) — an *original* route with zero vessels on it
#      right now (for instance, because Section 3 just moved its one vessel
#      onto a bypass) is still offered as a candidate edge as if a vessel
#      were coming. We additionally drop any edge whose route currently has
#      no deployed vessel at all, original or not, so a shipment isn't
#      routed onto a "ghost" service.
#   2. Missing bypass routes. Section 3's bypass routes set
#      ``source_service_route`` but never a matching ``disruption_key``, so
#      DefaultStrategy's own ``_route_is_available_for_booking`` silently
#      excludes them from candidate edges. We add them back in explicitly —
#      this is precisely what turns a "dynamically created bypass" into
#      something new cargo can actually be booked onto, instead of only
#      being useful to vessels already following it around.
#
# Net effect: a shipment whose shortest, disruption-free path would have
# gone through a route/leg that's temporarily out of service now
# transparently finds the next best viable option — another line, or one of
# this strategy's own bypasses — instead of stalling until DefaultStrategy's
# unmodified search happens to succeed on a later retry.

def _bypass_route_candidate_edges(congested_legs):
    """Candidate booking edges sourced only from this strategy's own
    Section-3 bypass routes, which DefaultStrategy's own candidate-edge
    builder silently excludes (see module docstring above)."""
    edges = []
    for route in _BYPASS_ROUTES.values():
        if not route.deployed_vessels:
            continue  # no vessel on it right now -> don't offer it as an option
        segments = sorted(route.segments, key=lambda s: s.sequence_index)
        segment_count = len(segments)
        for start_index in range(segment_count):
            departure_port = segments[start_index].associated_leg.departure_port
            cumulative_distance = 0.0
            for step in range(1, segment_count):
                segment_index = (start_index + step - 1) % segment_count
                leg = segments[segment_index].associated_leg
                cumulative_distance += leg.sailing_distance
                arrival_port = leg.arrival_port
                if departure_port == arrival_port:
                    continue
                candidate_segments = [
                    segments[(start_index + offset) % segment_count]
                    for offset in range(step)
                ]
                if any(seg.associated_leg in congested_legs for seg in candidate_segments):
                    continue
                edges.append(
                    _default_strategy._CandidateBookingEdge(
                        route, departure_port, arrival_port,
                        start_index + 1, segment_index + 1, cumulative_distance,
                    )
                )
    return edges


def _drop_suspended_route_edges(edges):
    """Remove candidate edges whose route currently has no deployed vessel."""
    return [edge for edge in edges if edge.service_route.deployed_vessels]


# =============================================================================
# Merged UserStrategy
# =============================================================================

class UserStrategy:
    @staticmethod
    @_safe_strategy(default_return=None)
    def select_vessel_for_berth(
        maritime_data_context,
        port,
        waiting_vessels,
        available_berths,
        current_time,
        waiting_since_by_vessel=None,
    ):
        """PortResponseStrategy — convoy-coordination + network-aware berthing.

        Convoys are still grouped purely by route similarity (JU, Section 1)
        — "vessels headed the same way" doesn't change. What changes is how
        a convoy's priority, and the pick *within* it, are scored: waiting
        time is now weighted by how clear each vessel's very next leg is
        (Section 4), using JP's own congestion signal
        (``_active_congestion_end``, Section 2). A vessel whose next leg is
        open is sent forward preferentially — it frees this berth fast and
        doesn't add to a downstream jam. A vessel whose next leg is
        severely congested with no known bypass is deprioritized — there is
        little point rushing it out of a safe berth only to have it queue
        up, or anchor, at the next bottleneck. A vessel with a *known*
        Section-3 bypass route around its bottleneck sits in between: it
        still has somewhere useful to go.

        The weighting is multiplicative on top of waiting time, not a hard
        veto — a vessel that has waited long enough still eventually wins a
        berth over a fresher, "clearer" one, which avoids starving a
        genuinely stuck vessel forever.

        Wrapped in ``@_safe_strategy(default_return=None)``: any unexpected
        exception degrades this single call to "no opinion", letting
        DefaultStrategy.select_vessel_for_berth decide instead, rather than
        aborting the whole run.
        """
        if not waiting_vessels:
            return None
        if len(waiting_vessels) == 1:
            return waiting_vessels[0]

        waiting_since_by_vessel = waiting_since_by_vessel or {}

        def waiting_hours(vessel):
            waiting_since = waiting_since_by_vessel.get(vessel, current_time)
            return max(0.0, (current_time - waiting_since).total_seconds() / 3600.0)

        def berthing_urgency(vessel):
            status = _vessel_next_leg_status(maritime_data_context, vessel, current_time)
            weight = _NEXT_LEG_PRIORITY_WEIGHT[status]
            # +1 hour floor so a vessel that just arrived (waiting_hours == 0)
            # can still express its next-leg status instead of scoring zero
            # regardless of how clear or blocked it is.
            return (waiting_hours(vessel) + 1.0) * weight

        ordered_vessels = sorted(waiting_vessels, key=waiting_hours, reverse=True)
        route_ports_by_vessel = {v: _route_ports(v) for v in waiting_vessels}

        mode = _congestion_mode(waiting_vessels, available_berths)
        max_size = _SMALL_CONVOY_MAX_SIZE if mode == "small" else None
        convoys = _build_convoys(ordered_vessels, route_ports_by_vessel, max_size)

        def convoy_urgency(convoy):
            return sum(berthing_urgency(v) for v in convoy)

        priority_convoy = max(convoys, key=convoy_urgency)
        return max(priority_convoy, key=berthing_urgency)

    @staticmethod
    @_safe_strategy(default_return=None)
    def create_alternative_service_routes(context, now, vessel=None):
        """ShippingLineResponseStrategy — Segment-Skipping bypass (see Section 3).

        Only acts when called with a specific ``vessel`` whose very next
        segment sits on a leg under active, high-multiplier disruption
        (checked with JP's own ``_active_congestion_end``). In that case it
        clones a one-off ServiceRoute that skips just that bottleneck port,
        using only a leg that already exists in ``context.legs``, and moves
        the vessel onto it -- but only while the vessel is empty, since a
        loaded vessel's Bookings still reference the original route (see
        Section 3 docstring for the full rationale), and only while the
        MAX_ACTIVE_BYPASSES_TOTAL / MAX_BYPASS_ROUTES_PER_SOURCE_ROUTE
        budgets are not exhausted.

        The origin-booking call site (``vessel=None``) always falls through
        to DefaultStrategy unchanged, and so does any vessel-specific call
        where no safe direct bypass exists -- returning ``None`` in every
        such case, exactly as the two contributors' original strategies did.

        Wrapped in ``@_safe_strategy(default_return=None)``: any unexpected
        exception degrades this call to "no opinion" instead of aborting the
        run; because the state-mutating calls happen only after every
        precondition (including the budget checks) has passed, an exception
        here should be rare, but the decorator is the last line of defense.
        """
        _restore_vessels_whose_bypass_cleared(context, now)

        if vessel is None:
            return None

        if vessel in _ACTIVE_BYPASS_BY_VESSEL:
            return None  # already bypassing this leg; let it run its course

        # Resilience: global cap on simultaneously-active bypasses.
        if len(_ACTIVE_BYPASS_BY_VESSEL) >= MAX_ACTIVE_BYPASSES_TOTAL:
            return None  # budget exhausted -> fall back to DefaultStrategy

        route = vessel.assigned_service_route
        if route is None or vessel.carried_shipments:
            return None  # only ever reroute an empty vessel

        next_segment = _peek_next_segment(vessel)
        if next_segment is None or next_segment.associated_leg is None:
            return None

        if _active_congestion_end(context, next_segment.associated_leg, now) is None:
            return None  # next leg isn't under an active, high-multiplier disruption

        bypass_route = _get_or_build_bypass_route(context, route, next_segment)
        if bypass_route is None:
            return None  # no viable direct bypass, or over budget -> fall back safely

        _detach_vessel_from_route(vessel, route)
        anchor_segment = max(bypass_route.segments, key=lambda s: s.sequence_index)
        _attach_vessel_to_route(vessel, bypass_route, anchor_segment)

        _ACTIVE_BYPASS_BY_VESSEL[vessel] = (route, bypass_route, next_segment.associated_leg)
        return True

    @staticmethod
    @_safe_strategy(default_return=False)
    def assign_associated_bookings(context, now, shipment):
        """CargoOwnerResponseStrategy — proactive rerouting (see Section 5).

        Builds the shipment's initial booking chain the same way
        DefaultStrategy does — shortest existing-leg path from origin to
        destination, avoiding actively-disrupted ports/legs — but the
        candidate-edge graph additionally (a) excludes any route, original
        or bypass, with zero deployed vessels right now, and (b) includes
        Section 3's bypass routes, which DefaultStrategy's own search never
        sees. That keeps a shipment from being handed a route that was
        just modified or effectively suspended, and lets it use a bypass
        this strategy already built instead of waiting for a slot on the
        original, congested service.

        Returns ``False`` (same contract as DefaultStrategy) when no path
        exists at all — the shipment keeps waiting at origin and is
        retried automatically once the active disruption's window closes.

        Wrapped in ``@_safe_strategy(default_return=False)`` — note the
        fallback is ``False``, not ``None``: it matches this method's own
        "no path found yet" contract (keep waiting, retry later) rather
        than the "delegate entirely to DefaultStrategy" contract used by
        the other three hooks.
        """
        demand = shipment.demand
        origin_port = demand.origin_port
        destination_port = demand.destination_port

        _default_strategy._remove_bookings_from_service_routes(shipment.associated_bookings)
        shipment.associated_bookings = []
        shipment.current_booking_index = None

        if origin_port == destination_port:
            return True

        avoid_port_names = OrderedSet()
        congested_legs = OrderedSet()
        if is_disruption_active(context, now):
            close_berth_plans, congested_leg_plans = _default_strategy._get_active_disruption_plans(context, now)
            avoid_port_names = _default_strategy._get_avoid_port_names(close_berth_plans)
            congested_legs = _default_strategy._get_congested_legs(congested_leg_plans)

        candidate_bookings = _default_strategy._build_all_candidate_bookings(
            context, avoid_port_names, congested_legs
        )
        candidate_bookings = _drop_suspended_route_edges(candidate_bookings)
        candidate_bookings += _bypass_route_candidate_edges(congested_legs)

        path = None
        if destination_port.name.casefold() not in avoid_port_names:
            path = _default_strategy._find_shortest_booking_path(
                context, origin_port, destination_port, candidate_bookings
            )

        if not path:
            return False

        for sequence_index, edge in enumerate(path, start=1):
            booking = Booking(
                sequence_index=sequence_index,
                shipment=shipment,
                service_route=edge.service_route,
                departure_segment_index=edge.departure_segment_index,
                arrival_segment_index=edge.arrival_segment_index,
            )
            shipment.associated_bookings.append(booking)
            edge.service_route.associated_bookings.append(booking)

        shipment.current_booking_index = min(
            booking.sequence_index for booking in shipment.associated_bookings
        )
        return True

    @staticmethod
    @_safe_strategy(default_return=None)
    def adjust_bookings_before_cargo_handling(context, now, vessel):
        """CargoOwnerResponseStrategy — hold-when-cheaper variant (JP).

        Hold cargo whose next leg is disrupted, when waiting is
        arithmetically cheaper than sailing through it now. Always
        returns None so DefaultStrategy still runs its own replanning
        afterwards — this only adds holding on top; it never replaces
        the default's rerouting.

        Holding is bounded by MAX_HELD_BOOKINGS_TOTAL /
        MAX_HELD_BOOKINGS_PER_PORT (see Section 2), so a single very broad
        or long-running set of disruptions cannot make the held-booking
        registry grow without limit.

        Previously this method carried its own local ``try/except
        Exception: pass``; that is now handled uniformly by the
        ``@_safe_strategy(default_return=None)`` decorator shared with the
        other three hooks, so all four have the exact same safety net and
        the same logged diagnostics on failure.
        """
        _reset_if_new_run(now)
        _release_due_holds(now)

        if not is_disruption_active(context, now):
            return None

        # Step 1: take cargo that is riding through off the vessel, so
        # step 2 can hold it. Runs before cargo handling, so setting
        # arrival_segment_index to the current segment makes it discharge here.
        _discharge_carried_cargo_that_should_wait(context, vessel, now)

        # Step 2: hold cargo stored at this port whose next leg is
        # worth waiting out (subject to the aggregate budgets above).
        port = _get_vessel_current_port(vessel)
        if port is not None:
            _hold_departures_that_should_wait(context, port, now)

        return None
