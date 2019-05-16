import random

from mesa import Agent, Model
from mesa.time import RandomActivation


class AirlineOne(Agent):
    """Airline 1 type"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.in_line = True
        self.at_stand = True
        self.time_at_stand = random.randint(20, 30)

    def step(self):
        print(f"plane {self.unique_id}: {self.at_stand} for {self.time_at_stand}")
        if self.at_stand:
            if self.time_at_stand > 0:
                self.time_at_stand -= 1
            else:
                # Remove the plane from the simulation
                print(f"Plane {self.unique_id} leaving stand")
                self.model.schedule.remove(self)


class Stand(object):
    """An airport stand"""
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.type_one_slots = [None, None, None]
        self.type_two_slots = [None, None, None, None]

    def __repr__(self):
        return f"<Stand {self.unique_id} type 1: {self.type_one_slots}, type 2: {self.type_two_slots}>"


class AirportModel(Model):
    def __init__(self, n):
        self.number_planes = n
        self.schedule = RandomActivation(self)
        self.stands = {i: Stand(i) for i in range(7)}

        # create planes
        for i in range(self.number_planes):
            plane = AirlineOne(i, self)
            self.schedule.add(plane)

    def step(self):
        self.schedule.step()

    def remove_plane_from_stand(self, plane):
        self.stands


if __name__ == '__main__':
    airport = AirportModel(10)
    print(airport)
    for _ in range(25):
        airport.step()
    print(f"{len(airport.schedule.agents)} planes left in stands")
    print(airport.stands)
