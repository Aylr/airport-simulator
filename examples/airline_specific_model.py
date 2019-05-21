"""
This is an Airport model with typed stands

Run it with `python airport_model_with_typed_stands.py`

If you like this can also be dropped into a jupyter notebook.
"""
from simulator.model import AirportModel

TICKS_TO_RUN_SIMULATION = 1000

print(f"Running simulation for {TICKS_TO_RUN_SIMULATION} ticks.")

airport = AirportModel(width=20, height=20, verbose=False)

for _ in range(TICKS_TO_RUN_SIMULATION):
    airport.step()

print("Simulation ended")

airport.plot_positions_heatmap()
airport.plot_position_history()

# Save data files as csvs with timestamps
airport.save_data_files()
