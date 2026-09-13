from scenario_builders import BaselineStableScenario


def test_baseline_stable_scenario_loads_csv_inputs():
    context = BaselineStableScenario.create()

    assert len(context.ports) == 20
    assert len(context.service_routes) == 9
    assert context.initial_service_routes == context.service_routes
    assert len(context.legs) == 54
    assert len(context.vessels) == 41
    assert sum(demand.annual_teus for demand in context.demands) > 0
