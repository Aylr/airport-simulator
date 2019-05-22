from simulator.server import build_server
from simulator.specific_model import AirportModel

server = build_server("Airline Specific Airport Model", AirportModel)
server.port = 8521  # The default
server.launch()
