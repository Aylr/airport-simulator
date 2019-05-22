# Airline Simulator

## Installation

1. If you are on a mac or linux system you can run `make install` and skip step 2
2. If you are not you need to run:
    - `pip install -r requirements.txt`
    - `pip install -r dev-requirements.txt`
    - `pip install -e ./`
3. Verify installation with `pip list` and look for the package called `simulator 1.x.x`
4. If you run `pytest` and see green you know installation succeeded

## Running Simulations

### Airline-specific models

- Run the example file with `python examples/specific_model.py`
    - You can edit the parameters like ticks to run, ratios, birth rates, etc there.
    - Data is logged to two csvs for further analysis.
- Run the interactive server with `python simulator/specific_run.py` which should open your browser

### Airline-agnostic models

- Run the example file with `python examples/agnostic_model.py`
    - You can edit the parameters like ticks to run, ratios, birth rates, etc there.
    - Data is logged to two csvs for further analysis.
- Run the interactive server with `python simulator/agnostic_run.py` which should open your browser
