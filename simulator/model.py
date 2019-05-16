import random
import numpy as np
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class AirlineOne(Agent):
    """Airline 1 type"""

    def __init__(self, unique_id, model, verbose=True):
        super().__init__(unique_id, model)
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

    def move(self):
        # TODO limit to non-diagonal movements
        # TODO limit to non-backwards movements
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        if self.verbose:
            print(f"moving plane from {self.pos} to {new_position}")
        self.model.grid.move_agent(self, new_position)


class Stand(object):
    """An airport stand"""

    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.type_one_slots = [None, None, None]
        self.type_two_slots = [None, None, None, None]

    def __repr__(self):
        return f"<Stand {self.unique_id} type 1: {self.type_one_slots}, type 2: {self.type_two_slots}>"


class AirportModel(Model):
    def __init__(self, n, width=20, height=20, verbose=True):
        self.number_planes = n
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        self.stands = {i: Stand(i) for i in range(7)}

        # create planes
        for i in range(self.number_planes):
            plane = AirlineOne(i, self, verbose=verbose)
            self.schedule.add(plane)
            self.grid.place_agent(plane, (0, 0))

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
        df = self.datacollector.get_agent_vars_dataframe()
        df.reset_index(inplace=True)

        for agent in df.AgentID.unique():
            x = df[df.AgentID == agent].x
            y = df[df.AgentID == agent].y
            plt.plot(x, y)

        plt.legend(df.AgentID)
        plt.show()


if __name__ == "__main__":
    airport = AirportModel(3, verbose=False)
    print(airport)
    for _ in range(25):
        airport.step()
    print(f"{len(airport.schedule.agents)} planes left in stands")
    print(airport.stands)
    airport.plot_position_history()

    airport.plot_positions()
    plt.show()
