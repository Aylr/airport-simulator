from simulator.server import build_server
from simulator.agnostic_model import AgnosticAirportModel

server = build_server("Airline Agnostic Airport Model", AgnosticAirportModel)
server.port = 8521  # The default
server.launch()
