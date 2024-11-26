from datetime import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import os
from time_function import timer

import pandas as pd


class NewTechnology:
    """Class used to test pandas version 1"""

    def __init__(self, scenario: int):
        """Initate the class used to test pandas version 1"""
        self.scenario = scenario
        self.save_location = (
            Path(os.getcwd()) / "data" / "data_files" / f"scenario_{scenario}"
        )

    @timer
    def load_data(self):
        """TASK 0: Load the data from the save location"""
        ...

    @timer
    def data_exploration(self):
        """TASK 1: Data exploration"""
        ...

    @timer
    def display_final_table(self):
        """TASK 2: Display the final table"""
        ...

    @timer
    def train_statsmodels_model(self):
        """TASK 3: Train a simple OLS model using Statsmodels"""
        ...

    @timer
    def train_sklearn_model(self):
        """TASK 4: Train a simple linear regression model using Scikit-learn"""
        ...

    def _get_final_table(self) -> pd.DataFrame:
        """Create the final table"""
        ...
