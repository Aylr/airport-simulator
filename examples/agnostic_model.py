"""
This is an Airport model with agnostic stands

Run it with `python agnostic_model.py`

If you like this can also be dropped into a jupyter notebook.
"""
from simulator.agnostic_model import AgnosticAirportModel

TICKS_TO_RUN_SIMULATION = 1000

print(f"Running simulation for {TICKS_TO_RUN_SIMULATION} ticks.")

airport = AgnosticAirportModel(width=20, height=20, verbose=False)

for _ in range(TICKS_TO_RUN_SIMULATION):
    airport.step()

print("Simulation ended")

airport.plot_positions_heatmap()
airport.plot_position_history()

# Save data files as csvs with timestamps
airport.save_data_files()
