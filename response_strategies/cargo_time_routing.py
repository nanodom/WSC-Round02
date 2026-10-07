"""Time-aware cargo routing (e-MPC) + WSPT berth selection.

Original author: the team's professor (proposed as a standalone user_strategy.py).
Kept verbatim; user_strategy.py delegates to it when PROF_CARGO_ENABLED is on,
and combines it with the fleet detour (create_alternative_service_routes).

Original docstring follows.

Strategy functions that contestants may modify.

Each function is called by the simulation at a specific decision point. The
``ShippingLineResponseStrategy`` and ``CargoOwnerResponseStrategy`` labels
describe responsibilities, but contestants do not have to keep their logic
strictly separated. When useful, the two response types may be combined. For
example, the logic for ``create_alternative_service_routes`` may instead be
implemented as part of ``adjust_bookings_before_cargo_handling`` so route and
vessel changes are decided together with shipment booking changes.

Implemented strategy (self-contained in this file)
---------------------------------------------------
* PortResponseStrategy -- weighted shortest processing time (WSPT): at a
  congested port, serve first the vessel with the most delayed TEU (on board
  plus waiting on the quay) per hour of handling, with a small aging term.
* CargoOwnerResponseStrategy -- time-aware routing inspired by event-triggered
  model predictive control for vessel schedule recovery (Zheng et al., 2024,
  Computers & Industrial Engineering 193, 110340):
    - booking paths minimise expected transport *hours* (service frequency,
      sailing time with the current congestion multiplier, port dwell, and the
      remaining closure time of ports still closed on arrival) instead of
      distance, because the KPI is TEU-weighted transport time;
    - by default only disruptions that are already active are used (their
      end time is known once they start); PARAMS["forecast_horizon_hours"]
      optionally lets the strategy also see plans starting within that
      horizon (experimental: check the competition rules before enabling);
    - every time a vessel reaches a port, carried shipments are re-planned
      (receding horizon) and switched only when the saving is significant.
* ShippingLineResponseStrategy -- not overridden; the default alternative
  service routes are kept.
"""

import datetime as dt
import heapq
import math
from dataclasses import dataclass

from maritime_data_context import Booking
from simulation_model.ordered_set import OrderedSet


class UserStrategy:
    @staticmethod
    def select_vessel_for_berth(
        maritime_data_context,
        port,
        waiting_vessels,
        available_berths,
        current_time,
        waiting_since_by_vessel=None,
    ):
        """PortResponseStrategy.

        Select the next vessel to receive a berth at a congested port.

        This function is called when the number of waiting vessels reaches the
        configured port-congestion threshold. It is not called when the normal
        first-in-first-out selection is sufficient.

        Parameters
        ----------
        maritime_data_context:
            The complete maritime data context.
        port:
            The ``Port`` where a berth is being assigned.
        waiting_vessels:
            Ordered list of vessels currently waiting at ``port``. The selected
            vessel must be an object from this list.
        available_berths:
            List of currently available berth objects at ``port``. This can be
            used to inspect available capacity, but this function selects a
            vessel rather than a berth.
        current_time:
            Current simulation time as a ``datetime``.
        waiting_since_by_vessel:
            Mapping ``{vessel: waiting_start_time}``. It may be ``None``.
            Use it to calculate how long each vessel has waited.

        Returns
        -------
        Vessel
            Return exactly one vessel contained in ``waiting_vessels``.
            Returning another object raises a ``ValueError``.
        """
        return _select_vessel_wspt(
            port, waiting_vessels, current_time, waiting_since_by_vessel
        )

    @staticmethod
    def create_alternative_service_routes(context, now, vessel=None):
        """ShippingLineResponseStrategy.

        Optionally create disruption-avoiding routes from existing legs and
        reserve existing vessels for those routes.

        A newly created service route must be composed only of ``Leg`` objects
        that already exist in ``context.legs``. This strategy must not create
        new legs. Vessels assigned to a new route must be transferred from
        existing service routes; this strategy must not create new vessels, and
        the total number of vessels in ``context.vessels`` must remain unchanged.

        The simulation validates these constraints after every call, including
        calls that return ``None``. Returning ``None`` means the method did not
        handle the decision and must leave the context unchanged so the default
        implementation can run safely.

        Return ``None`` to use the default implementation. This logic may
        instead be incorporated into ``adjust_bookings_before_cargo_handling``
        when a combined shipping-line and cargo-owner decision is preferred.
        """
        return None

    @staticmethod
    def assign_associated_bookings(context, now, shipment):
        """CargoOwnerResponseStrategy.

        Assign the initial booking chain for a newly generated shipment.

        A custom strategy should create the required ``Booking`` objects,
        populate ``shipment.associated_bookings`` in sequence order, register
        each booking in its service route's ``associated_bookings`` collection,
        and set ``shipment.current_booking_index``.

        Parameters
        ----------
        context:
            The complete maritime data context. The shipment's origin and
            destination are available through ``shipment.demand``.
        now:
            Current simulation time as a ``datetime``.
        shipment:
            The ``Shipment`` that needs its initial bookings.

        Returns
        -------
        bool
            Return ``True`` when a valid booking chain has been assigned.
            Return ``False`` when no booking can currently be assigned; the
            simulation may keep the shipment waiting and retry later.
        """
        return _assign_time_aware_bookings(context, now, shipment)

    @staticmethod
    def adjust_bookings_before_cargo_handling(context, now, vessel):
        """CargoOwnerResponseStrategy.

        Replan carried shipments before a vessel starts cargo handling.

        This is the only in-transit booking-replanning decision point. It is
        called after the vessel reaches a port and before loading and discharging
        decisions are processed. A custom strategy may inspect
        ``vessel.carried_shipments`` and modify each affected shipment's booking
        chain and current booking index.

        Parameters
        ----------
        context:
            The complete maritime data context, including active
            ``disruption_plans`` and available ``service_routes``.
        now:
            Current simulation time as a ``datetime``.
        vessel:
            The arriving ``Vessel``. Its current location is represented by
            ``vessel.current_segment`` and its onboard shipments by
            ``vessel.carried_shipments``.

        Returns
        -------
        bool
            Return ``True`` after updating the affected booking chains.
        """
        return _replan_carried_shipments(context, now, vessel)


# ---------------------------------------------------------------------------
# Tunable parameters (hours unless stated otherwise).
# ---------------------------------------------------------------------------
PARAMS = {
    # Expected berth time per port call (handling plus pilotage).
    "port_dwell_hours": 12.0,
    # Extra handling/storage time charged for every transshipment.
    "transfer_penalty_hours": 12.0,
    # Extra hours charged after a closed port reopens (backlog at the berths).
    "reopen_backlog_hours": 24.0,
    # Fraction of the route headway used as expected boarding wait.
    "headway_wait_fraction": 0.5,
    # Use real vessel positions to estimate the first boarding wait.
    "use_vessel_eta_for_first_boarding": True,
    # Re-plan carried shipments only when the saving exceeds both limits.
    "replan_min_saving_hours": 24.0,
    "replan_min_saving_fraction": 0.10,
    # Re-plan carried shipments even when no disruption is active.
    "replan_without_disruption": False,
    # How far ahead (hours) planned disruptions are taken into account before
    # they start. 0 = react only to active disruptions.
    "forecast_horizon_hours": 0.0,
    # Reuse a computed origin-destination path for this many simulated hours
    # while the network state (disruptions, vessel deployment) is unchanged.
    "route_cache_hours": 6.0,
}

_PROFILE_CACHE = {}
_PATH_CACHE = {}
_CACHE_OWNER = [None]


def _bind_cache(context):
    """Drop cached routes when a new simulation context starts."""
    if _CACHE_OWNER[0] is not context:
        _PROFILE_CACHE.clear()
        _PATH_CACHE.clear()
        _CACHE_OWNER[0] = context


def _hours(delta):
    return delta.total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Disruption snapshot: active plans, plus plans starting within the forecast
# horizon when PARAMS["forecast_horizon_hours"] > 0.
# ---------------------------------------------------------------------------
class _DisruptionSnapshot:
    """Known port closures and leg congestions as windows relative to now.

    Every window is ``(start_hours, end_hours)`` measured from ``now``; an
    active disruption has ``start_hours <= 0``.
    """

    def __init__(self, context, now):
        horizon = max(0.0, PARAMS["forecast_horizon_hours"])
        berth_windows = {}
        self.leg_windows = {}
        for plan in context.disruption_plans:
            if plan.start_offset_days is None or plan.duration_days is None:
                continue
            start = dt.datetime.min + dt.timedelta(days=plan.start_offset_days)
            end = start + dt.timedelta(days=plan.duration_days)
            start_hours = _hours(start - now)
            end_hours = _hours(end - now)
            if end_hours <= 0.0 or start_hours > horizon:
                continue
            window = (start_hours, end_hours)
            if plan.target_berth is not None and plan.close_berth:
                port = plan.target_berth.port
                berth_windows.setdefault((port, window), OrderedSet()).add(
                    plan.target_berth
                )
            if plan.target_leg is not None and plan.multiplier > 1.0:
                self.leg_windows.setdefault(plan.target_leg, []).append(
                    (start_hours, end_hours, plan.multiplier)
                )

        # A port is blocked only while all of its berths are closed.
        self.port_windows = {}
        for (port, window), berths in berth_windows.items():
            if len(berths) >= len(port.berths):
                self.port_windows.setdefault(port, []).append(window)
        self.is_active = bool(self.port_windows or self.leg_windows)

    def leg_hours(self, leg, speed, depart_at):
        base = leg.sailing_distance / speed
        for start_hours, end_hours, multiplier in self.leg_windows.get(leg, ()):
            if start_hours <= depart_at < end_hours:
                return base * multiplier
        return base

    def port_wait_hours(self, port, arrive_at):
        wait = 0.0
        for start_hours, end_hours in self.port_windows.get(port, ()):
            if start_hours <= arrive_at < end_hours:
                wait = max(
                    wait, end_hours - arrive_at + PARAMS["reopen_backlog_hours"]
                )
        return wait


# ---------------------------------------------------------------------------
# Route profiles.
# ---------------------------------------------------------------------------
class _RouteProfile:
    def __init__(self, route):
        self.route = route
        self.segments = sorted(route.segments, key=lambda s: s.sequence_index)
        vessels = list(route.deployed_vessels)
        speeds = [
            v.vessel_class.sailing_speed
            for v in vessels
            if v.vessel_class is not None and v.vessel_class.sailing_speed > 0
        ]
        self.speed = sum(speeds) / len(speeds) if speeds else 0.0
        self.vessel_count = len(vessels)
        if self.speed > 0 and self.segments:
            cycle = sum(
                s.associated_leg.sailing_distance / self.speed
                + PARAMS["port_dwell_hours"]
                for s in self.segments
            )
        else:
            cycle = math.inf
        self.cycle_hours = cycle
        self.headway_hours = (
            cycle / self.vessel_count if self.vessel_count else math.inf
        )
        self.starts_by_port = {}
        for position, segment in enumerate(self.segments):
            port = segment.associated_leg.departure_port
            self.starts_by_port.setdefault(port, []).append(position)

    @property
    def usable(self):
        return self.vessel_count > 0 and self.speed > 0 and len(self.segments) > 1

    def boarding_wait(self, port, position, now_offset, use_eta):
        """Expected hours until a vessel of this route departs ``port``."""
        base = PARAMS["headway_wait_fraction"] * self.headway_hours
        if not use_eta or now_offset > 0.0:
            return base
        eta = self._next_vessel_eta(position)
        return base if eta is None else min(base * 2.0, eta)

    def _next_vessel_eta(self, target_position):
        """Rough hours until the next vessel reaches the target segment start."""
        count = len(self.segments)
        position_by_segment = {
            id(segment): index for index, segment in enumerate(self.segments)
        }
        best = None
        for vessel in self.route.deployed_vessels:
            current = vessel.current_segment
            if current is None or id(current) not in position_by_segment:
                continue
            index = position_by_segment[id(current)]
            # The vessel is on (or at the end of) segment ``index``; assume it is
            # half way through that leg, then add the following legs.
            leg = self.segments[index].associated_leg
            hours = 0.5 * leg.sailing_distance / self.speed
            cursor = (index + 1) % count
            steps = 0
            while cursor != target_position and steps < count:
                hours += PARAMS["port_dwell_hours"]
                hours += self.segments[cursor].associated_leg.sailing_distance / self.speed
                cursor = (cursor + 1) % count
                steps += 1
            hours += PARAMS["port_dwell_hours"]
            if best is None or hours < best:
                best = hours
        return best


def _network_key(context, now):
    _bind_cache(context)
    return (
        id(context),
        _get_active_disruption_key(context, now),
        tuple(
            (id(route), len(route.deployed_vessels))
            for route in context.service_routes
        ),
    )


def _available_routes(context, now, network_key=None):
    network_key = network_key or _network_key(context, now)
    cached = _PROFILE_CACHE.get(id(context))
    if cached is not None and cached[0] == network_key:
        return cached[1]
    active_key = network_key[1]
    profiles = []
    for route in context.service_routes:
        if route.source_service_route is not None and route.disruption_key != active_key:
            continue
        profile = _RouteProfile(route)
        if profile.usable:
            profiles.append(profile)
    _PROFILE_CACHE[id(context)] = (network_key, profiles)
    return profiles


# ---------------------------------------------------------------------------
# Time-dependent shortest path over booking edges.
# ---------------------------------------------------------------------------
def _walk_route(profile, snapshot, start_position, depart_at, max_steps=None):
    """Yield (arrival_port, arrival_time, segment) along the route."""
    count = len(profile.segments)
    time_cursor = depart_at
    steps = count - 1 if max_steps is None else max_steps
    for step in range(steps):
        segment = profile.segments[(start_position + step) % count]
        leg = segment.associated_leg
        time_cursor += snapshot.leg_hours(leg, profile.speed, time_cursor)
        time_cursor += snapshot.port_wait_hours(leg.arrival_port, time_cursor)
        yield leg.arrival_port, time_cursor, segment
        time_cursor += PARAMS["port_dwell_hours"]


def _shortest_booking_path(
    profiles, snapshot, origin_port, destination_port, stay_edges=None
):
    """Return (hours, [edges]) minimising expected arrival time."""
    use_eta = PARAMS["use_vessel_eta_for_first_boarding"]
    best = {origin_port: 0.0}
    previous = {}
    heap = [(0.0, 0, origin_port)]
    counter = 1
    visited = OrderedSet()

    while heap:
        time_now, _, port = heapq.heappop(heap)
        if port in visited:
            continue
        visited.add(port)
        if port is destination_port:
            break

        candidates = []
        if port is origin_port and stay_edges:
            candidates.extend(stay_edges)
        for profile in profiles:
            for position in profile.starts_by_port.get(port, ()):
                if (
                    port is origin_port
                    and stay_edges
                    and profile.route is stay_edges[0][0].route
                ):
                    # Boarding a later vessel of the same route from here is
                    # never better than staying on the current one.
                    continue
                wait = profile.boarding_wait(port, position, time_now, use_eta)
                is_transfer = port is not origin_port or bool(stay_edges)
                transfer = PARAMS["transfer_penalty_hours"] if is_transfer else 0.0
                candidates.append((profile, position, wait + transfer))

        for profile, position, wait in candidates:
            depart_at = time_now + wait
            departure_segment = profile.segments[position]
            distance = 0.0
            for arrival_port, arrival_time, segment in _walk_route(
                profile, snapshot, position, depart_at
            ):
                distance += segment.associated_leg.sailing_distance
                if arrival_port is port:
                    break
                if arrival_time < best.get(arrival_port, math.inf):
                    best[arrival_port] = arrival_time
                    previous[arrival_port] = (
                        port,
                        _CandidateBookingEdge(
                            profile.route,
                            port,
                            arrival_port,
                            departure_segment.sequence_index,
                            segment.sequence_index,
                            distance,
                        ),
                    )
                    heapq.heappush(heap, (arrival_time, counter, arrival_port))
                    counter += 1

    if destination_port not in previous:
        return math.inf, None
    path = []
    cursor = destination_port
    while cursor is not origin_port:
        parent, edge = previous[cursor]
        path.append(edge)
        cursor = parent
    path.reverse()
    return best[destination_port], path


def _profile_for(profiles, route):
    return next((p for p in profiles if p.route is route), None)


def _position_of(profile, sequence_index):
    for position, segment in enumerate(profile.segments):
        if segment.sequence_index == sequence_index:
            return position
    return -1


def _steps_between(profile, start_position, end_position):
    count = len(profile.segments)
    return ((end_position - start_position) % count) + 1


# ---------------------------------------------------------------------------
# Cargo-owner decisions: initial booking and in-transit re-planning.
# ---------------------------------------------------------------------------
def _assign_time_aware_bookings(context, now, shipment):
    demand = shipment.demand
    origin_port = demand.origin_port
    destination_port = demand.destination_port

    _remove_bookings_from_service_routes(shipment.associated_bookings)
    shipment.associated_bookings = []
    shipment.current_booking_index = None

    if origin_port == destination_port:
        return True

    network_key = _network_key(context, now)
    cache_key = (id(context), origin_port, destination_port)
    cached = _PATH_CACHE.get(cache_key)
    if cached is not None and cached[0] == network_key and now < cached[1]:
        path = cached[2]
    else:
        profiles = _available_routes(context, now, network_key)
        snapshot = _DisruptionSnapshot(context, now)
        hours, path = _shortest_booking_path(
            profiles, snapshot, origin_port, destination_port
        )
        if math.isinf(hours):
            path = None
        expires = now + dt.timedelta(hours=PARAMS["route_cache_hours"])
        _PATH_CACHE[cache_key] = (network_key, expires, path)
    if not path:
        return False

    for index, edge in enumerate(path, start=1):
        booking = Booking(
            sequence_index=index,
            shipment=shipment,
            service_route=edge.service_route,
            departure_segment_index=edge.departure_segment_index,
            arrival_segment_index=edge.arrival_segment_index,
        )
        shipment.associated_bookings.append(booking)
        edge.service_route.associated_bookings.append(booking)
    shipment.current_booking_index = 1
    return True


def _replan_carried_shipments(context, now, vessel):
    snapshot = _DisruptionSnapshot(context, now)
    if not snapshot.is_active and not PARAMS["replan_without_disruption"]:
        return

    current_segment = vessel.current_segment
    route = vessel.assigned_service_route
    if current_segment is None or route is None or not vessel.carried_shipments:
        return
    current_port = current_segment.associated_leg.arrival_port

    profiles = _available_routes(context, now)
    vessel_profile = _profile_for(profiles, route) or _RouteProfile(route)
    if vessel_profile.speed <= 0:
        return
    current_position = _position_of(vessel_profile, current_segment.sequence_index)
    if current_position < 0:
        return
    next_position = (current_position + 1) % len(vessel_profile.segments)
    stay_edges = [(vessel_profile, next_position, PARAMS["port_dwell_hours"])]

    best_cache = {}
    for shipment in list(vessel.carried_shipments):
        try:
            current_booking = shipment.get_current_booking()
        except ValueError:
            continue
        if current_booking.service_route is not route:
            continue
        final_port = _get_final_booking_port(shipment)
        if final_port is None or final_port is current_port:
            continue

        keep_hours = _remaining_plan_hours(
            profiles,
            snapshot,
            shipment,
            current_booking,
            vessel_profile,
            current_position,
        )

        key = final_port
        if key not in best_cache:
            best_cache[key] = _shortest_booking_path(
                profiles, snapshot, current_port, final_port, stay_edges
            )
        new_hours, path = best_cache[key]
        if not path or math.isinf(new_hours):
            continue

        saving = keep_hours - new_hours
        threshold = max(
            PARAMS["replan_min_saving_hours"],
            PARAMS["replan_min_saving_fraction"] * min(keep_hours, 1e9),
        )
        if not math.isinf(keep_hours) and saving < threshold:
            continue

        _replace_unfinished_bookings_from_current_port(
            shipment, current_booking, current_segment, path
        )
    return True


def _remaining_plan_hours(
    profiles, snapshot, shipment, current_booking, vessel_profile, current_position
):
    """Expected hours to finish the shipment's current booking chain.

    Uses the same time accounting as ``_shortest_booking_path`` so the
    keep-versus-replan comparison is fair.
    """
    use_eta = PARAMS["use_vessel_eta_for_first_boarding"]
    arrival_position = _position_of(
        vessel_profile, current_booking.arrival_segment_index
    )
    if arrival_position < 0:
        return math.inf
    time_cursor = 0.0
    if arrival_position != current_position:
        time_cursor = PARAMS["port_dwell_hours"]
        start = (current_position + 1) % len(vessel_profile.segments)
        steps = _steps_between(vessel_profile, start, arrival_position)
        for _, arrival_time, _ in _walk_route(
            vessel_profile, snapshot, start, time_cursor, max_steps=steps
        ):
            time_cursor = arrival_time

    later = sorted(
        (
            b
            for b in shipment.associated_bookings
            if b.sequence_index > current_booking.sequence_index
        ),
        key=lambda b: b.sequence_index,
    )
    for booking in later:
        profile = _profile_for(profiles, booking.service_route)
        if profile is None:
            return math.inf
        start = _position_of(profile, booking.departure_segment_index)
        end = _position_of(profile, booking.arrival_segment_index)
        if start < 0 or end < 0:
            return math.inf
        port = profile.segments[start].associated_leg.departure_port
        time_cursor += profile.boarding_wait(port, start, time_cursor, use_eta)
        time_cursor += PARAMS["transfer_penalty_hours"]
        steps = _steps_between(profile, start, end)
        for _, arrival_time, _ in _walk_route(
            profile, snapshot, start, time_cursor, max_steps=steps
        ):
            time_cursor = arrival_time
    return time_cursor


# ---------------------------------------------------------------------------
# Booking-graph helpers (own copies so this file does not depend on the
# private functions of default_strategy.py).
# ---------------------------------------------------------------------------
@dataclass
class _CandidateBookingEdge:
    service_route: object
    departure_port: object
    arrival_port: object
    departure_segment_index: int
    arrival_segment_index: int
    total_distance: float


def _leg_key(leg):
    return (
        leg.departure_port.name.casefold(),
        leg.arrival_port.name.casefold(),
    )


def _is_active(plan, now):
    if plan.start_offset_days is None or plan.duration_days is None:
        return False
    start = dt.datetime.min + dt.timedelta(days=plan.start_offset_days)
    end = start + dt.timedelta(days=plan.duration_days)
    return start <= now < end


def _get_active_disruption_key(context, now):
    """Same key the default strategy stores on its alternative routes."""
    avoid_port_names = OrderedSet()
    congested_leg_keys = OrderedSet()
    for plan in context.disruption_plans:
        if not _is_active(plan, now):
            continue
        if plan.close_berth and plan.target_berth is not None:
            avoid_port_names.add(plan.target_berth.port.name.casefold())
        if plan.multiplier > 1 and plan.target_leg is not None:
            congested_leg_keys.add(_leg_key(plan.target_leg))
    return (
        tuple(sorted(avoid_port_names)),
        tuple(sorted(congested_leg_keys)),
    )


def _remove_bookings_from_service_routes(bookings):
    """Remove stale reverse references for bookings no longer owned by a shipment."""
    for booking in bookings:
        service_route = booking.service_route
        if service_route is None:
            continue
        while booking in service_route.associated_bookings:
            service_route.associated_bookings.remove(booking)


def _get_final_booking_port(shipment):
    last_booking = max(
        shipment.associated_bookings, key=lambda b: b.sequence_index, default=None
    )
    if last_booking is None or last_booking.service_route is None:
        return None
    final_segment = next(
        (
            segment
            for segment in last_booking.service_route.segments
            if segment.sequence_index == last_booking.arrival_segment_index
        ),
        None,
    )
    return final_segment.associated_leg.arrival_port if final_segment else None


def _replace_unfinished_bookings_from_current_port(
    shipment, current_booking, current_segment, path
):
    """Cut the current booking at this port and append the new path.

    When the first new edge continues on the same route (the vessel the
    shipment is already on), it is merged into the current booking so the
    shipment simply stays on board.
    """
    original_bookings = list(shipment.associated_bookings)
    retained = sorted(
        (
            booking
            for booking in shipment.associated_bookings
            if booking.sequence_index < current_booking.sequence_index
        ),
        key=lambda booking: booking.sequence_index,
    )

    completed_booking = current_booking
    completed_booking.arrival_segment_index = current_segment.sequence_index

    next_sequence = current_booking.sequence_index + 1
    new_bookings = []
    first_edge = path[0]
    if first_edge.service_route is completed_booking.service_route:
        completed_booking.arrival_segment_index = first_edge.arrival_segment_index
        remaining_edges = path[1:]
    else:
        remaining_edges = path

    for edge in remaining_edges:
        new_bookings.append(
            Booking(
                sequence_index=next_sequence,
                shipment=shipment,
                service_route=edge.service_route,
                departure_segment_index=edge.departure_segment_index,
                arrival_segment_index=edge.arrival_segment_index,
            )
        )
        next_sequence += 1

    retained_bookings = retained + [completed_booking]
    replaced_bookings = [
        booking for booking in original_bookings if booking not in retained_bookings
    ]
    _remove_bookings_from_service_routes(replaced_bookings)
    for booking in new_bookings:
        booking.service_route.associated_bookings.append(booking)
    shipment.associated_bookings.clear()
    shipment.associated_bookings.extend(retained_bookings + new_bookings)
    shipment.current_booking_index = completed_booking.sequence_index


# ---------------------------------------------------------------------------
# Port decision: weighted shortest processing time (WSPT).
# The KPI is TEU-weighted transport time, so while a vessel waits every TEU on
# board and every TEU waiting on the quay for it is delayed. Serving vessels in
# decreasing order of (affected TEU / handling hours) minimises TEU-weighted
# waiting at a congested port; a small aging term prevents starvation.
# ---------------------------------------------------------------------------
# Productivity used by the simulation model: 45 TEU/hour per quay crane,
# roughly one crane per 50 metres of vessel length.
_TEU_PER_CRANE_HOUR = 45.0
_MIN_HANDLING_HOURS = 1.0
# Priority boost per hour already waited, relative to the mean priority.
_AGING_PER_HOUR = 0.01


def _select_vessel_wspt(port, waiting_vessels, current_time, waiting_since_by_vessel):
    if not waiting_vessels:
        return None
    waiting_since_by_vessel = waiting_since_by_vessel or {}

    scores = []
    for vessel in waiting_vessels:
        on_board, discharge, load = _teu_profile(vessel, port)
        cranes = max(1, round((vessel.vessel_class.loa or 0) / 50))
        handling = max(
            _MIN_HANDLING_HOURS, (discharge + load) / (cranes * _TEU_PER_CRANE_HOUR)
        )
        scores.append((on_board + load) / handling)

    mean_score = sum(scores) / len(scores) or 1.0
    best_index = 0
    best_value = -math.inf
    for index, vessel in enumerate(waiting_vessels):
        since = waiting_since_by_vessel.get(vessel, current_time)
        waited = max(0.0, (current_time - since).total_seconds() / 3600.0)
        value = scores[index] + _AGING_PER_HOUR * waited * mean_score
        if value > best_value:
            best_value = value
            best_index = index
    return waiting_vessels[best_index]


def _teu_profile(vessel, port):
    """Return (TEU on board, TEU to discharge here, TEU to load here)."""
    on_board = 0
    discharge = 0
    current = vessel.current_segment
    for shipment in vessel.carried_shipments:
        size = shipment.teu_size or 0
        on_board += size
        try:
            booking = shipment.get_current_booking()
        except ValueError:
            continue
        if (
            current is not None
            and booking.service_route is vessel.assigned_service_route
            and booking.arrival_segment_index == current.sequence_index
        ):
            discharge += size

    load = 0
    route = vessel.assigned_service_route
    if route is not None:
        try:
            next_segment = vessel.get_next_segment()
        except ValueError:
            next_segment = None
        if next_segment is not None:
            for shipment in port.shipments_in_storage:
                try:
                    booking = shipment.get_current_booking()
                except ValueError:
                    continue
                if (
                    booking.service_route is route
                    and booking.departure_segment_index == next_segment.sequence_index
                ):
                    load += shipment.teu_size or 0
    capacity = vessel.vessel_class.teu_capacity if vessel.vessel_class else load
    return on_board, discharge, min(load, capacity)
