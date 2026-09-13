from maritime_data_context import Demand, MaritimeDataContext, Port
from simulation_model.shipment_generator import ShipmentGenerator


def test_zero_teu_generation_opportunity_does_not_create_shipment(monkeypatch):
    context = MaritimeDataContext()
    origin = Port(name="Origin")
    destination = Port(name="Destination")
    demand = Demand(origin, destination, annual_teus=1)
    context.demands.append(demand)
    origin.outgoing_demands.append(demand)
    destination.incoming_demands.append(demand)

    generator = ShipmentGenerator(context, seed=1)
    monkeypatch.setattr(generator, "_sample_shipment_teu_size", lambda expected: 0)
    monkeypatch.setattr(generator, "_next_interarrival_time", lambda: 1.0)

    generator._generate(demand)

    assert demand.shipments == []
    assert origin.shipments_in_storage == []
    assert generator._next_shipment_index == 1
