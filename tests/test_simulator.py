import pytest

from simulator.model import Airline, AirlineStates, AirportModel


def test_default_airline():
    plane = Airline(1, "foo")
    assert plane.airline_type in [1, 2]
    assert plane.state == AirlineStates.IN_LINE
    assert plane.time_at_stand <= 30
    assert plane.tick_count == 0


def test_default_airport_model():
    airport = AirportModel()
    assert airport.line == []
    assert airport.max_plane_id == 0
    assert len(airport.stands) == 7

def test_nothing():
    assert True
