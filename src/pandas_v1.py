import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import os
from time_function import timer

import pandas as pd


class PandasV1:
    """Class used to test pandas version 1"""

    def __init__(self, scenario: int):
        """Initate the class used to test pandas version 1"""
        self.scenario = scenario
        self.save_location = Path(os.getcwd()) / "data" / "data_files" / f"scenario_{scenario}"

    @timer
    def load_data(self):
        """TASK 0: Load the data from the save location"""
        customer_database = pd.read_parquet(self.save_location / "customer_database.parquet")
        customer_database["customer_id"] = customer_database["customer_id"].astype(str)
        customer_database["brand_happiness"] = customer_database["brand_happiness"].astype(float)
        self.customer_database = customer_database

        store_visits = pd.read_parquet(self.save_location / "store_visits.parquet")
        store_visits["customer_id"] = store_visits["customer_id"].astype(str)
        store_visits["datetime"] = pd.to_datetime(store_visits["datetime"])
        self.store_visits = store_visits

        advertisements = pd.read_parquet(self.save_location / "advertisements.parquet")
        advertisements["datetime"] = pd.to_datetime(advertisements["datetime"])
        advertisements["channel"] = advertisements["channel"].astype(str)
        advertisements["spend"] = advertisements["spend"].astype(float)
        self.advertisements = advertisements

        revenue = pd.read_parquet(self.save_location / "revenue.parquet")
        revenue["datetime"] = pd.to_datetime(revenue["datetime"])
        revenue["revenue"] = revenue["revenue"].astype(float)
        self.revenue = revenue

    @timer
    def data_exploration(self):
        """TASK 1: Data exploration"""
        # Print the number of NaN values in brand_happiness
        num_nans = self.customer_database["brand_happiness"].isna().sum()
        print(f"Number of NaN values in brand_happiness: {num_nans}\n")

        # Display the top 10 customers by visit count
        visit_counts = self.store_visits["customer_id"].value_counts()

        print("Top 10 customers")
        print(visit_counts.head(10))

        # Group by channel and calculate the average and sum of spend
        average_spend_per_channel = self.advertisements.groupby("channel")["spend"].mean()
        sum_spend_per_channel = self.advertisements.groupby("channel")["spend"].sum()

        # Print the results
        print("\nAverage spend per channel:")
        print(average_spend_per_channel)
        print("\nSum of spend per channel:")
        print(sum_spend_per_channel)

    @timer
    def display_final_table(self):
        """TASK 2: Display the final table"""
        final_table = self._get_final_table()
        print(final_table)

    @timer
    def train_statsmodels_model(self):
        """TASK 3: Train a simple OLS model using Statsmodels"""
        final_table = self._get_final_table()

        X = final_table[["brand_happiness", "mixed", "offline", "online"]]
        y = final_table["revenue"]
        X = sm.add_constant(X)
        statsmodels_model = sm.OLS(y, X).fit()
        print(statsmodels_model.summary())

    @timer
    def train_sklearn_model(self):
        """TASK 4: Train a simple linear regression model using Scikit-learn"""
        final_table = self._get_final_table()

        X = final_table[["brand_happiness", "mixed", "offline", "online"]]
        X.columns = X.columns.astype(str)

        y = final_table["revenue"]

        # Initialize and fit the linear regression model
        statsmodels_model = LinearRegression()
        statsmodels_model.fit(X, y)

        # Print the coefficients and intercept
        print("Coefficients:", statsmodels_model.coef_)
        print("Intercept:", statsmodels_model.intercept_)

        # Print the R squared value
        y_pred = statsmodels_model.predict(X)
        print("R-squared:", r2_score(y, y_pred))

    def _get_final_table(self) -> pd.DataFrame:
        """Create the final table"""
        daily_advertisements = self.advertisements.groupby(["datetime", "channel"]).agg({"spend": "sum"}).reset_index()
        daily_advertisements = daily_advertisements.pivot(
            index="datetime", columns="channel", values="spend"
        ).reset_index()

        # Match the customers with their brand happiness and do some data cleaning
        daily_visits = self.store_visits.merge(self.customer_database, on="customer_id")
        daily_visits = daily_visits.dropna(subset=["brand_happiness"])
        daily_visits = daily_visits[(daily_visits["brand_happiness"] <= 100) & (daily_visits["brand_happiness"] >= 1)]

        avg_daily_brand_happiness = daily_visits.groupby("datetime").agg({"brand_happiness": "mean"}).reset_index()

        # Merge all the data together
        final_table = daily_advertisements.merge(avg_daily_brand_happiness, on="datetime")
        final_table = final_table.merge(self.revenue, on="datetime")
        return final_table
