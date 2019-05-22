import random

from simulator.specific_model import Airline, AirportModel, AirlineStates


class AgnosticAirline(Airline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_open_stands_at_same_y(self):
        """Build a list of open stands ignoring type at the same Y coordinate."""
        return [
            s
            for id, s in self.model.stands.items()
            if s.y == self.y_position and s.is_occupied is False
        ]

    def get_planes_at_same_y(self, taxiing_planes):
        """Get planes on same Y."""
        planes_at_same_y = [
            p for p in taxiing_planes if p != self and p.y_position == self.y_position
        ]
        return planes_at_same_y


class AgnosticAirportModel(AirportModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_plane_to_line(self):
        """Add a plane to the simulation and start it in line."""
        if random.random() < self.type_1_ratio:
            airline_type = 1
        else:
            airline_type = 2

        plane = AgnosticAirline(
            unique_id=self.max_plane_id,
            model=self,
            airline_type=airline_type,
            min_stand_time=self.min_stand_time,
            max_stand_time=self.max_stand_time,
            verbose=self.verbose,
        )
        self.max_plane_id += 1
        self.schedule.add(plane)
        self.grid.place_agent(plane, (0, 0))
        self.line.append(plane)

    def can_first_plane_in_line_begin_taxiing(self):
        """
        Check if the first plane in line can begiun taxiing.
        """
        if not self.line:
            # If there is no line, skip this
            return False

        planes_taxiing = self.get_planes_in_state(AirlineStates.TAXIING_TO_STAND)
        stands = self.stands.values()
        empty_stands = [s for s in stands if s.is_occupied is False]

        return len(empty_stands) > len(planes_taxiing)

    def get_stands_of_type(self, airline_type):
        """Get all stands of a given airline type."""
        return [s for id, s in self.stands.items()]

    def get_open_stands(self, airline_type):
        """Get all open stands of a given airline type."""
        return [s for s in self.get_stands_of_type() if not s.is_occupied]

    @property
    def model_type(self):
        return "airline-agnostic"
