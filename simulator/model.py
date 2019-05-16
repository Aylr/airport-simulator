import random
import numpy as np
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class Airline(Agent):
    def __init__(self, unique_id, model, verbose=True):
        super().__init__(unique_id, model)
        self.airline_type = random.choice([1, 2])
        self.in_line = True
        self.at_stand = False
        self.time_at_stand = random.randint(20, 30)
        self.verbose = verbose

    @property
    def x_position(self):
        return self.pos[0]

    @property
    def y_position(self):
        return self.pos[1]

    def step(self):
        if self.verbose:
            print(f"plane {self.unique_id}: {self.at_stand} for {self.time_at_stand}")
        if self.at_stand:
            if self.time_at_stand > 0:
                self.time_at_stand -= 1
            else:
                # Remove the plane from the simulation
                if self.verbose:
                    print(f"Plane {self.unique_id} leaving stand")
                self.model.schedule.remove(self)
        else:
            self.move()

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
        if not isinstance(stands, list):
            stands = [stands]
        deltas = []
        for stand in stands:
            delta_x = abs(stand.x - self.x_position)
            delta_y = abs(stand.y - self.y_position)
            deltas.append(delta_x + delta_y)

        min_indices = [i for i, x in enumerate(deltas) if x == min(deltas)]
        return [stands[i] for i in min_indices]

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # TODO move towards open stand
        # TODO move towards closest open stand
        open_stands = self.model.get_open_stands(self.airline_type)
        closest_stands = self.closest_stands(open_stands)

        picked_stand = random.choice(closest_stands)
        steps_closer = self.get_moves_closer(possible_steps, picked_stand)
        if steps_closer:
            new_position = self.random.choice(steps_closer)
            if self.verbose:
                print(f"moving plane from {self.pos} to {new_position}")
            self.model.grid.move_agent(self, new_position)


class Stand(object):
    """An airport stand"""

    def __init__(self, unique_id, airline_type, x, y):
        self.unique_id = unique_id
        self.airline_type = airline_type
        self.x = x
        self.y = y
        self.is_occupied = None

    def __repr__(self):
        return f"<Stand {self.unique_id} accepts type: {self.airline_type} slot: {self.is_occupied}>"


class AirportModel(Model):
    def __init__(self, n, width=20, height=20, verbose=True):
        self.number_planes = n
        self.schedule = RandomActivation(self)
        # TODO raise error on smallest grid
        self.grid = MultiGrid(width, height, torus=False)
        self.stands = {
            1: Stand(1, 1, 1, height - 4),
            2: Stand(2, 1, 1, height - 6),
            3: Stand(3, 1, 1, height - 8),
            4: Stand(4, 2, width - 1, height - 4),
            5: Stand(5, 2, width - 1, height - 6),
            6: Stand(6, 2, width - 1, height - 8),
            7: Stand(7, 2, width - 1, height - 10),
        }

        # create planes
        for i in range(self.number_planes):
            plane = Airline(i, self, verbose=verbose)
            self.schedule.add(plane)
            # TODO middle of grid
            self.grid.place_agent(plane, (10, 0))

        self.datacollector = DataCollector(
            model_reporters={"number_planes": "number_planes"},
            agent_reporters={"x": "x_position", "y": "y_position"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def remove_plane_from_stand(self, plane):
        pass

    def add_plane_to_stand(self):
        pass

    def get_stands_of_type(self, airline_type):
        return [s for id, s in self.stands.items() if s.airline_type == airline_type]

    def get_open_stands(self, airline_type):
        return [s for s in self.get_stands_of_type(airline_type) if not s.is_occupied]

    def plot_positions(self):
        agent_counts = np.zeros((self.grid.width, self.grid.height))
        for cell in self.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(cell_content)
            agent_counts[x][y] = agent_count
        plt.imshow(agent_counts, interpolation="nearest")
        plt.colorbar()
        # If running from a text editor or IDE, remember you'll need the following:
        plt.show()

    def plot_position_history(self):
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


if __name__ == "__main__":
    airport = AirportModel(10, verbose=False)
    print(airport)
    for _ in range(100):
        airport.step()
    print(f"{len(airport.schedule.agents)} planes left in stands")
    print(airport.stands)

    # airport.plot_positions()
    airport.plot_position_history()
    # plt.show()
