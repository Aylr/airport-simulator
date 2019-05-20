# server.py
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from simulator.model import AirportModel, Airline, Stand, AirlineStates


def agent_portrayal(agent):
    if isinstance(agent, Airline):
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 0,
            "Color": "red",
            "r": 0.75,
            "text": agent.unique_id,
            "text_color": "white",
        }

        if agent.airline_type == 1:
            portrayal["Color"] = "blue"
        else:
            portrayal["Color"] = "green"

        if agent.state == AirlineStates.IN_LINE:
            portrayal["text_color"] = "red"
        elif agent.state == AirlineStates.TAXIING_TO_STAND:
            portrayal["text_color"] = "white"

        if agent.is_at_stand:
            portrayal["r"] = (0.75 / 30) * agent.unloading_time_when_at_stand
        return portrayal
    elif isinstance(agent, Stand):
        portrayal = {
            "Shape": "circle",
            "Layer": 0,
            "r": 1,
            "text": agent.unique_id,
            "text_color": "white",
        }

        if agent.airline_type == 1:
            portrayal["Color"] = "blue"
        else:
            portrayal["Color"] = "green"

        return portrayal


birth_rate = UserSettableParameter(
    "slider", "Plane Birth Rate Per Tick", value=0.5, min_value=0, max_value=1, step=0.1
)

chart = ChartModule(
    [
        {"Label": "planes_in_line", "Color": "Red"},
        {"Label": "planes_taxiing_to_stand", "Color": "Orange"},
        {"Label": "planes_at_stand", "Color": "Green"},
        {"Label": "planes_served_at_stand_1", "Color": "Green"},
        {"Label": "planes_served_at_stand_2", "Color": "Green"},
        {"Label": "planes_served_at_stand_3", "Color": "Green"},
        {"Label": "planes_served_at_stand_4", "Color": "Green"},
        {"Label": "planes_served_at_stand_5", "Color": "Green"},
        {"Label": "planes_served_at_stand_6", "Color": "Green"},
        {"Label": "planes_served_at_stand_7", "Color": "Green"},
    ],
    data_collector_name="datacollector",
)

grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
server = ModularServer(
    AirportModel,
    [grid, chart],
    "Airport Model",
    {"width": 20, "height": 20, "birth_rate": birth_rate},
)
