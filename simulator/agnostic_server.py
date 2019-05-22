from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from simulator.agnostic_model import AgnosticAirportModel, AgnosticAirline
from simulator.airline_specific_model import Stand, AirlineStates


def agent_portrayal(agent):
    if isinstance(agent, AgnosticAirline):
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
            "Color": "Grey",
        }

        return portrayal


birth_rate = UserSettableParameter(
    "slider", "Plane Birth Rate Per Tick", value=0.5, min_value=0, max_value=1, step=0.1
)

minimum_stand_time = UserSettableParameter(
    "slider", "Minimum Stand Time", value=20, min_value=1, max_value=100, step=1
)

maximum_stand_time = UserSettableParameter(
    "slider", "Maximum Stand Time", value=30, min_value=1, max_value=100, step=1
)

type_1_ratio = UserSettableParameter(
    "slider", "Airline Type 1 ratio", value=0.5, min_value=0, max_value=1, step=0.01
)

plane_state_chart = ChartModule(
    [
        {"Label": "number_of_planes_taxiing_to_stand", "Color": "Orange"},
        {"Label": "number_of_planes_at_stand", "Color": "Red"},
    ],
    data_collector_name="datacollector",
)


grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

display_elements = [grid, plane_state_chart]

for chart_details in [
    {"Label": "number_of_planes_in_line", "Color": "Red"},
    {"Label": "planes_served_at_stand_1", "Color": "Orange"},
    {"Label": "planes_served_at_stand_2", "Color": "Yellow"},
    {"Label": "planes_served_at_stand_3", "Color": "Green"},
    {"Label": "planes_served_at_stand_4", "Color": "Blue"},
    {"Label": "planes_served_at_stand_5", "Color": "Indigo"},
    {"Label": "planes_served_at_stand_6", "Color": "Violet"},
    {"Label": "planes_served_at_stand_7", "Color": "Black"},
]:
    display_elements.append(
        ChartModule([chart_details], data_collector_name="datacollector")
    )


server = ModularServer(
    AgnosticAirportModel,
    display_elements,
    "Airline Agnostic Airport Model",
    {
        "width": 20,
        "height": 20,
        "birth_rate": birth_rate,
        "type_1_ratio": type_1_ratio,
        "min_stand_time": minimum_stand_time,
        "max_stand_time": maximum_stand_time,
    },
)
