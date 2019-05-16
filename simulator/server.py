# server.py
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from model import AirportModel


def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "Color": "red",
        "r": 0.75,
        "text": agent.unique_id,
        "text_color": 'white'
    }

    if agent.airline_type == 1:
        portrayal["Color"] = "blue"
    else:
        portrayal["Color"] = "green"

    if agent.is_at_stand:
        portrayal["r"] = (0.75 / 30) * agent.time_at_stand
    return portrayal


grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
server = ModularServer(
    AirportModel, [grid], "Airport Model", {"width": 20, "height": 20}
)
