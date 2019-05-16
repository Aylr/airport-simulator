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
        self.time_at_stand = random.randint(20, 30)
        self.verbose = verbose
        self.tick_count = 0

    @property
    def x_position(self):
        return self.pos[0]

    @property
    def y_position(self):
        return self.pos[1]

    def step(self):
        self.tick_count += 1

        if self.verbose:
            print(f"plane {self.unique_id}: {self.is_at_stand} for {self.time_at_stand}")
        if self.is_at_stand:
            # TODO this isn't super efficient to do this every step
            self.model.add_plane_to_stand(self)

            if self.time_at_stand <= 0:
                # Remove the plane from the simulation
                if self.verbose:
                    print(f"Plane {self.unique_id} leaving stand")
                self.model.remove_plane(self)

            self.time_at_stand -= 1
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

    @property
    def is_at_stand(self):
        for id, stand in self.model.stands.items():
            if self.pos == stand.position:
                return True

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        open_stands = self.model.get_open_stands(self.airline_type)
        closest_stands = self.closest_stands(open_stands)

        if closest_stands:
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
        self.is_occupied = False

    @property
    def position(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"<Stand {self.unique_id} accepts type: {self.airline_type} is_occupied: {self.is_occupied}>"


class AirportModel(Model):
    def __init__(self, n, width=20, height=20, verbose=True):
        # TODO probability of adding plane on a tick
        self.number_planes = n
        self.width = width
        self.max_plane_id = 0
        self.verbose = verbose
        self.schedule = RandomActivation(self)
        # TODO raise error on smallest grid
        self.grid = MultiGrid(width, height, torus=False)
        # TODO space out stands more dynamically
        self.stands = {
            1: Stand(1, 1, 1, height - 4),
            2: Stand(2, 1, 1, height - 6),
            3: Stand(3, 1, 1, height - 8),
            4: Stand(4, 2, width - 1, height - 4),
            5: Stand(5, 2, width - 1, height - 6),
            6: Stand(6, 2, width - 1, height - 8),
            7: Stand(7, 2, width - 1, height - 10),
        }

        self.datacollector = DataCollector(
            # TODO stand metrics
            # TODO planes served
            # TODO planes queued
            # TODO queue duration
            model_reporters={"number_planes": "number_planes", },
            agent_reporters={
                "tick_count": "tick_count",
                "x": "x_position",
                "y": "y_position",
                "is_at_stand": "is_at_stand",
                "time_at_stand": "time_at_stand",
            },
        )

    def add_plane(self):
        plane = Airline(self.max_plane_id, self, verbose=self.verbose)
        self.max_plane_id += 1

        self.schedule.add(plane)
        middle_x = int(self.width / 2)
        self.grid.place_agent(plane, (middle_x, 0))

    def step(self):
        if random.random() >= 0.7:
            self.add_plane()

        self.datacollector.collect(self)
        self.schedule.step()

    def remove_plane(self, plane):
        for id, stand in self.stands.items():
            if plane.pos == stand.position:
                self.stands[id].is_occupied = False
                break
        self.schedule.remove(plane)

    def add_plane_to_stand(self, plane):
        for id, stand in self.stands.items():
            if plane.pos == stand.position:
                self.stands[id].is_occupied = True
                print(f"Plane {plane.unique_id} docked at stand {id}")
                print(self.stands)
                break

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
    airport = AirportModel(6, width=20, height=20, verbose=False)
    print(airport)
    for _ in range(50):
        airport.step()
    print(f"{len(airport.schedule.agents)} planes left in stands")
    print(airport.stands)

    # airport.plot_positions()
    airport.plot_position_history()
    # plt.show()
    df = airport.datacollector.get_agent_vars_dataframe()
