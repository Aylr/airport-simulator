# Airline Simulator

## Installation

1. If you are on a mac or linux system you can run `make install`
2. If you are not you can run:
    - `pip install -r requirements.txt`
    - `pip install -r dev-requirements.txt`
    - `pip install -e ./setup.py`
3. Verify installation with `pip list` and look for the package called `simulator 1.0.0`
3. If you run `pytest` and see green you know installation succeeded

## Running

- Run the example file with `python examples/airport_model_with_typed_stands.py`
    - Data is logged to two csvs for further analysis.
- Run the server with `python simulator/run.py` which should open your browser
