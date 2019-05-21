import pytest

from simulator.model import Airline, AirlineStates, AirportModel


def test_default_airline():
    plane = Airline(1, "foo", 1)
    assert plane.airline_type in [1, 2]
    assert plane.state == AirlineStates.IN_LINE
    assert plane.unloading_time_when_at_stand <= 30
    assert plane.tick_count == 0


def test_default_airport_model():
    airport = AirportModel()
    assert airport.line == []
    assert airport.max_plane_id == 0
    assert len(airport.stands) == 7
