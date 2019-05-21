from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# This constant represents the two kinds of airlines and stands
AIRLINE_TYPES = [1, 2]


class AirlineStates(object):
    """
    These are the 3 states a plan can be in:

    1. In line. It starts it's life here.
    2. Taxiing towards a stand.
    3. Parked at the stand loading and unloading.
    4. Vanished from the simulation.
    """

    IN_LINE = "in_line"
    TAXIING_TO_STAND = "taxiing_to_stand"
    AT_STAND = "at_stand"
    OFF_SIMULATION = "off_simulation"


class Airline(Agent):
    """
    This is the plane agent.
    """

    def __init__(
        self,
        unique_id,
        model,
        airline_type,
        min_stand_time=20,
        max_stand_time=30,
        verbose=True,
    ):
        super().__init__(unique_id, model)
        self.airline_type = airline_type
        self.state = AirlineStates.IN_LINE

        self.unloading_time_when_at_stand = random.randint(
            min_stand_time, max_stand_time
        )
        self.verbose = verbose
        self.tick_count = 0

    @property
    def x_position(self):
        return self.pos[0]

    @property
    def y_position(self):
        return self.pos[1]

    def step(self):
        if self.verbose:
            print(f"plane {self.unique_id}: {self.state}")

        self.tick_count += 1

        if self.state == AirlineStates.IN_LINE:
            # Do nothing else this tick.
            pass
        elif self.state == AirlineStates.TAXIING_TO_STAND:
            self.move()
        elif self.state == AirlineStates.AT_STAND:
            self.unloading_time_when_at_stand -= 1

    def get_moves_closer(self, possible_steps, stand):
        """Filter potential moves using manhattan distance."""
        results = []
        current_delta_x = abs(stand.x - self.x_position)
        current_delta_y = abs(stand.y - self.y_position)

        for step in possible_steps:
            x, y = step
            delta_x = abs(stand.x - x)
            delta_y = abs(stand.y - y)
            if delta_x < current_delta_x or delta_y < current_delta_y:
                results.append(step)

        return results

    def closest_stands(self, stands):
        """get a list of the closest stand(s) via Manhattan distance."""
        deltas = []
        for stand in stands:
            delta_x = abs(stand.x - self.x_position)
            delta_y = abs(stand.y - self.y_position)
            deltas.append(delta_x + delta_y)

        min_indices = [i for i, x in enumerate(deltas) if x == min(deltas)]
        return [stands[i] for i in min_indices]

    def x_distance_from_closest_stand(self):
        stands_at_same_y = self.get_open_stands_at_same_y()
        closest_stand = self.closest_stands(stands_at_same_y)
        if closest_stand:
            return abs(self.x_position - closest_stand[0].x)

    @property
    def is_at_stand(self):
        # TODO this isn't super efficient.
        for id, stand in self.model.stands.items():
            if self.pos == stand.position:
                return True

    def move(self):
        possible_next_positions = self.get_possible_next_positions()
        open_stands_at_same_y = self.get_open_stands_at_same_y()
        taxiing_planes = self.model.get_planes_in_state(AirlineStates.TAXIING_TO_STAND)
        planes_at_same_y = self.get_planes_at_same_y(taxiing_planes)
        closer_planes_on_same_y = self.get_planes_on_same_y_closer_to_stand(
            planes_at_same_y
        )

        if not open_stands_at_same_y:
            # If there is not an open stand on this Y, then go up.
            self.move_plane_up_if_possible(possible_next_positions)
        elif closer_planes_on_same_y:
            # If there is an open stand on this Y, but a plane is closer, then go up
            self.move_plane_up_if_possible(possible_next_positions)

        elif open_stands_at_same_y:
            # If there is an open stand then move towards it
            closest_stands = self.closest_stands(open_stands_at_same_y)

            if closest_stands:
                picked_stand = random.choice(closest_stands)
                steps_closer = self.get_moves_closer(
                    possible_next_positions, picked_stand
                )

                if steps_closer:
                    # TODO if plane position did not change last tick, allow non-closer movements to jitter out of local minima
                    new_position = self.random.choice(steps_closer)
                    if self.verbose:
                        print(f"moving plane from {self.pos} to {new_position}")
                    self.model.grid.move_agent(self, new_position)

    def get_planes_on_same_y_closer_to_stand(self, planes_at_same_y):
        """Get planes on same Y that are closer to a stand than this one."""
        results = []
        x_distance_to_closest_stand = self.x_distance_from_closest_stand()

        if x_distance_to_closest_stand:
            for other_plane in planes_at_same_y:
                other_plane_x_distance_from_stand = (
                    other_plane.x_distance_from_closest_stand()
                )
                if (
                    other_plane_x_distance_from_stand
                    and other_plane_x_distance_from_stand < x_distance_to_closest_stand
                ):
                    results.append(other_plane)

        return results

    def get_planes_at_same_y(self, taxiing_planes):
        """Get planes on same Y."""
        planes_at_same_y = [
            p
            for p in taxiing_planes
            if p != self
            and p.y_position == self.y_position
            and p.airline_type == self.airline_type
        ]
        return planes_at_same_y

    def move_plane_up_if_possible(self, possible_next_positions):
        """Move plane up if the next position is empty."""
        cell_above_current_position = (self.x_position, self.y_position + 1)
        if cell_above_current_position in possible_next_positions:
            self.model.grid.move_agent(self, cell_above_current_position)

    def get_open_stands_at_same_y(self):
        """Build a list of open stands of the matching type at the same Y coordinate."""
        return [
            s
            for id, s in self.model.stands.items()
            if s.y == self.y_position
            and s.is_occupied is False
            and s.airline_type == self.airline_type
        ]

    def get_possible_next_positions(self):
        """Get possible next positions that are not occupied by a plane."""
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # Only allow movements to empty cells to avoid collisions
        empty_positions = [
            p for p in possible_steps if not self.model.is_plane_in_cell(p)
        ]
        return empty_positions


class Stand(Agent):
    """An airport stand"""

    def __init__(self, unique_id, airline_type, x, y, model):
        super().__init__(unique_id, model)
        assert (
            airline_type in AIRLINE_TYPES
        ), f"Stand must be one of type {AIRLINE_TYPES}"
        self.unique_id = unique_id
        self.airline_type = airline_type
        self.x = x
        self.y = y
        self.is_occupied = False
        self.planes_unloaded = []
        # TODO does this need a notion of which plane is docked here?

    @property
    def position(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"<Stand {self.unique_id} accepts type: {self.airline_type} is_occupied: {self.is_occupied}>"


class AirportModel(Model):
    """This represents an airport."""

    def __init__(
        self,
        width=20,
        height=20,
        birth_rate=0.1,
        type_1_ratio=0.75,
        min_stand_time=20,
        max_stand_time=30,
        verbose=False,
    ):
        super().__init__()
        self.width = width
        self.birth_rate = birth_rate
        self.type_1_ratio = type_1_ratio
        self.min_stand_time = min_stand_time
        self.max_stand_time = max_stand_time

        # This list holds the planes waiting on the tarmac before they can taxi to an open stand
        self.line = []
        self.max_plane_id = 0
        self.verbose = verbose
        self.schedule = RandomActivation(self)
        # TODO raise error on smallest grid
        self.grid = MultiGrid(width, height, torus=False)

        self.stands = {
            1: Stand(1, 1, 0, height - 4, model=self),
            2: Stand(2, 1, 0, height - 6, model=self),
            3: Stand(3, 1, 0, height - 8, model=self),
            4: Stand(4, 2, width - 1, height - 4, model=self),
            5: Stand(5, 2, width - 1, height - 6, model=self),
            6: Stand(6, 2, width - 1, height - 8, model=self),
            7: Stand(7, 2, width - 1, height - 10, model=self),
        }
        for id, stand in self.stands.items():
            self.grid.place_agent(stand, stand.position)

        self.datacollector = DataCollector(
            model_reporters={
                "number_of_planes_in_line": "number_of_planes_in_line",
                "number_of_planes_taxiing_to_stand": "number_of_planes_taxiing_to_stand",
                "number_of_planes_at_stand": "number_of_planes_at_stand",
                "planes_served_at_stand_1": "planes_served_at_stand_1",
                "planes_served_at_stand_2": "planes_served_at_stand_2",
                "planes_served_at_stand_3": "planes_served_at_stand_3",
                "planes_served_at_stand_4": "planes_served_at_stand_4",
                "planes_served_at_stand_5": "planes_served_at_stand_5",
                "planes_served_at_stand_6": "planes_served_at_stand_6",
                "planes_served_at_stand_7": "planes_served_at_stand_7",
            },
            agent_reporters={
                "tick_count": "tick_count",
                "state": "state",
                "x": "x_position",
                "y": "y_position",
                "is_at_stand": "is_at_stand",
                "unloading_time_when_at_stand": "unloading_time_when_at_stand",
            },
        )

    def add_plane_to_line(self):
        """Add a plane to the simulation and start it in line."""
        if random.random() < self.type_1_ratio:
            airline_type = 1
        else:
            airline_type = 2

        plane = Airline(
            unique_id=self.max_plane_id,
            model=self,
            airline_type=airline_type,
            min_stand_time=self.min_stand_time,
            max_stand_time=self.max_stand_time,
            verbose=self.verbose,
        )
        self.max_plane_id += 1

        self.schedule.add(plane)
        middle_x = int(self.width / 2)
        self.grid.place_agent(plane, (middle_x, 0))
        self.line.append(plane)

    def get_planes_in_state(self, state):
        """Get a list of all planes in a given AirlineState"""
        all_planes = self.schedule.agents
        planes_in_state = [p for p in all_planes if p.state == state]
        return planes_in_state

    def count_planes_in_state(self, state):
        return len(self.get_planes_in_state(state))

    def is_plane_in_cell(self, pos):
        """Is a plane in a given (x, y) cell."""
        contents = self.grid.get_cell_list_contents(pos)
        for thing in contents:
            if isinstance(thing, Airline):
                return True
        return False

    def is_stand_in_cell(self, pos):
        """Is a stand in a given (x, y) cell."""
        contents = self.grid.get_cell_list_contents(pos)
        for thing in contents:
            if isinstance(thing, Stand):
                return True
        return False

    @property
    def number_of_planes_in_line(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return self.count_planes_in_state(AirlineStates.IN_LINE)

    @property
    def number_of_planes_taxiing_to_stand(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return self.count_planes_in_state(AirlineStates.TAXIING_TO_STAND)

    @property
    def number_of_planes_at_stand(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return self.count_planes_in_state(AirlineStates.AT_STAND)

    @property
    def planes_served_at_stand_1(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[1].planes_unloaded)

    @property
    def planes_served_at_stand_2(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[2].planes_unloaded)

    @property
    def planes_served_at_stand_3(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[3].planes_unloaded)

    @property
    def planes_served_at_stand_4(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[4].planes_unloaded)

    @property
    def planes_served_at_stand_5(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[5].planes_unloaded)

    @property
    def planes_served_at_stand_6(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[6].planes_unloaded)

    @property
    def planes_served_at_stand_7(self):
        """Mesa doesn't allow a function with args to be in a datacollector, so this helper property exists."""
        return len(self.stands[7].planes_unloaded)

    def release_first_plane_in_line(self):
        """Release the first plane in line!"""
        first_plane_ine_line = self.line[0]
        first_plane_ine_line.state = AirlineStates.TAXIING_TO_STAND
        # remove plane from beginning of line (represented by index 0)
        self.line.pop(0)

    def can_first_plane_in_line_begin_taxiing(self):
        """
        Check if the first plane in line can begiun taxiing.

        First check if there is room on the tarmac for another plane to leave
        the line. This means that there are only the same number of planes of
        type 1 for the same amount of stands.
        For example, if there are 3 type 1 stands, then only ever allow 3 type
        1 planes out of the line.
        """
        # TODO this is where the model could be airline agnostic
        # TODO subclass AirportModel and override this method
        if len(self.line) == 0:
            # If there is no line, skip this
            return False

        result = False

        planes_taxiing = self.get_planes_in_state(AirlineStates.TAXIING_TO_STAND)
        type_1_planes_taxiing = [p for p in planes_taxiing if p.airline_type == 1]
        type_2_planes_taxiing = [p for p in planes_taxiing if p.airline_type == 2]

        stands = self.stands.values()
        type_1_empty_stands = [
            s for s in stands if s.airline_type == 1 and s.is_occupied is False
        ]
        type_2_empty_stands = [
            s for s in stands if s.airline_type == 2 and s.is_occupied is False
        ]

        type_1_ok_to_release = len(type_1_empty_stands) > len(type_1_planes_taxiing)
        type_2_ok_to_release = len(type_2_empty_stands) > len(type_2_planes_taxiing)

        if type_1_ok_to_release and type_2_ok_to_release:
            result = True

        return result

    def step(self):
        """Do all the important things during a tick like changing plane and stand states."""
        # This is the main state machine
        self.datacollector.collect(self)
        if random.random() <= self.birth_rate:
            self.add_plane_to_line()
        self.check_planes_in_line()
        self.check_planes_arriving_at_stands()
        self.check_planes_at_stands()
        self.schedule.step()

    def check_planes_in_line(self):
        """Release a plane from the line if it can be."""
        if self.can_first_plane_in_line_begin_taxiing():
            self.release_first_plane_in_line()

    def check_planes_at_stands(self):
        """Remove plane from the simulation if it is done at the stand."""
        for plane in self.schedule.agents:
            if (
                plane.state == AirlineStates.AT_STAND
                and plane.unloading_time_when_at_stand <= 0
            ):
                self.remove_plane(plane)
                if self.verbose:
                    print(f"Plane {plane.unique_id} leaving stand")

    def remove_plane(self, plane):
        """Do the dirty work of removing a plane from the simulation and grid."""
        # TODO this isn't super efficient
        for id, stand in self.stands.items():
            if plane.pos == stand.position:
                self.stands[id].is_occupied = False
                self.stands[id].planes_unloaded.append(plane.unique_id)
                break

        # remove the agent from the scheduler and the grid
        plane.state = AirlineStates.OFF_SIMULATION
        self.schedule.remove(plane)
        self.grid.remove_agent(plane)

    def get_active_planes_not_in_line(self):
        taxiing_planes = self.get_planes_in_state(AirlineStates.TAXIING_TO_STAND)
        planes_at_stands = self.get_planes_in_state(AirlineStates.AT_STAND)
        results = taxiing_planes + planes_at_stands
        assert isinstance(results, list)
        return results

    def check_planes_arriving_at_stands(self):
        """Check taxiing planes for state updates."""
        for id, stand in self.stands.items():
            for plane in self.get_active_planes_not_in_line():
                if plane.pos == stand.position:
                    self.stands[id].is_occupied = True
                    plane.state = AirlineStates.AT_STAND

    def get_stands_of_type(self, airline_type):
        """Get all stands of a given airline type."""
        return [s for id, s in self.stands.items() if s.airline_type == airline_type]

    def get_open_stands(self, airline_type):
        """Get all open stands of a given airline type."""
        return [s for s in self.get_stands_of_type(airline_type) if not s.is_occupied]

    def plot_positions_heatmap(self):
        agent_counts = np.zeros((self.grid.width, self.grid.height))
        for cell in self.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(
                [agent for agent in cell_content if isinstance(agent, Airline)]
            )
            agent_counts[x][y] = agent_count
        plt.imshow(agent_counts, interpolation="nearest")
        plt.colorbar()
        # If running from a text editor or IDE, remember you'll need the following:
        plt.show()
        return agent_counts

    def plot_position_history(self):
        """Plot tracks, which aren't super interesting for lots of ticks."""
        # plot stands
        for id, stand in self.stands.items():
            color = "r" if stand.airline_type == 1 else "b"
            plt.scatter(stand.x, stand.y, marker=".", color=color)

        # plot planes
        df = self.datacollector.get_agent_vars_dataframe()
        df.reset_index(inplace=True)

        for agent in df.AgentID.unique():
            x = df[df.AgentID == agent].x
            y = df[df.AgentID == agent].y
            plt.plot(x, y)

        # plt.legend(df.AgentID)
        plt.show()

    def save_data_files(self):
        """Save the model and agent datacollector dataframes as csvs."""
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()

        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d--%H-%M-%S")
        model_data.to_csv(f"{timestamp}_model_data.csv")
        agent_data.to_csv(f"{timestamp}_agent_data.csv")
        print(f"Saved files with timestamp {timestamp}")
