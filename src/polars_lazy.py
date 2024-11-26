import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import os
from time_function import timer

import polars as pl


class PolarsLazy:
    """Class used to test pandas version 1"""

    def __init__(self, scenario: int):
        """Initate the class used to test pandas version 1"""
        self.scenario = scenario
        self.save_location = Path(os.getcwd()) / "data" / "data_files" / f"scenario_{scenario}"

    @timer
    def load_data(self):
        """TASK 0: Load the data from the save location"""
        customer_database = pl.read_parquet(self.save_location / f"customer_database.parquet").lazy()
        customer_database = customer_database.select(
            pl.col("customer_id").cast(pl.String),
            pl.col("brand_happiness").cast(pl.Float64),
        )
        self.customer_database = customer_database

        store_visits = pl.read_parquet(self.save_location / f"store_visits.parquet").lazy()
        store_visits = store_visits.select(
            pl.col("customer_id").cast(pl.String),
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M"),
        )
        self.store_visits = store_visits

        advertisements = pl.read_parquet(self.save_location / f"advertisements.parquet").lazy()
        advertisements = advertisements.select(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M"),
            pl.col("channel").cast(pl.String),
            pl.col("spend").cast(pl.Float32),
        )
        self.advertisements = advertisements

        revenue = pl.read_parquet(self.save_location / f"revenue.parquet").lazy()
        revenue = revenue.select(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M"),
            pl.col("revenue").cast(pl.Float32),
        )
        self.revenue = revenue

    @timer
    def data_exploration(self):
        """TASK 1: Data exploration"""
        # Print the number of NaN values in brand_happiness
        num_nans = self.customer_database.select(pl.col("brand_happiness").is_nan().sum()).collect()
        print(f"Number of NaN values in brand_happiness: {num_nans}\n")

        # Display the top 10 customers by visit count
        visit_counts = self.store_visits.group_by("customer_id").len(name="total_visits")
        top_10_customers = visit_counts.top_k(10, by="total_visits")

        print("Top 10 customers")
        print(top_10_customers.collect())

        # Group by channel and calculate the average and sum of spend
        average_spend_per_channel = self.advertisements.group_by("channel").agg(
            pl.col("spend").mean(),
        )
        sum_spend_per_channel = self.advertisements.group_by("channel").agg(
            pl.col("spend").sum(),
        )

        # Print the results
        print("\nAverage spend per channel:")
        print(average_spend_per_channel.collect())
        print("\nSum of spend per channel:")
        print(sum_spend_per_channel.collect())

    @timer
    def display_final_table(self):
        """TASK 2: Display the final table"""
        final_table = self._get_final_table()
        print(final_table.collect())

    @timer
    def train_statsmodels_model(self):
        """TASK 3: Train a simple OLS model using Statsmodels"""
        final_table = self._get_final_table()
        final_table = final_table.collect()

        X = final_table.select("brand_happiness", "mixed", "offline", "online").to_numpy()
        y = final_table.select("revenue").to_numpy()
        X = sm.add_constant(X)
        statsmodels_model = sm.OLS(y, X).fit()
        print(statsmodels_model.summary())

    @timer
    def train_sklearn_model(self):
        """TASK 4: Train a simple linear regression model using Scikit-learn"""
        final_table = self._get_final_table()
        final_table = final_table.collect()

        X = final_table.select("brand_happiness", "mixed", "offline", "online")
        y = final_table.select("revenue")

        # Initialize and fit the linear regression model
        statsmodels_model = LinearRegression()
        statsmodels_model.fit(X, y)

        # Print the coefficients and intercept
        print("Coefficients:", statsmodels_model.coef_)
        print("Intercept:", statsmodels_model.intercept_)

        # Print the R squared value
        y_pred = statsmodels_model.predict(X)
        print("R-squared:", r2_score(y, y_pred))

    def _get_final_table(self) -> pl.LazyFrame:
        """Create the final table"""
        # Aggregate per channel and pivot so that each channel is a column
        advertisements_pivot = self.advertisements.with_columns(
            online=pl.when(pl.col("channel") == "online").then(pl.col("spend")).otherwise(None),
            offline=pl.when(pl.col("channel") == "offline").then(pl.col("spend")).otherwise(None),
            mixed=pl.when(pl.col("channel") == "mixed").then(pl.col("spend")).otherwise(None),
        ).drop("channel")

        daily_advertisements = advertisements_pivot.group_by("datetime").agg(
            pl.col("online").sum(),
            pl.col("offline").sum(),
            pl.col("mixed").sum(),
        )

        # Match the customers with their brand happiness and do some data cleaning
        daily_visits = self.store_visits.join(self.customer_database, on="customer_id")
        daily_visits = daily_visits.filter(pl.col("brand_happiness").is_not_nan())
        daily_visits = daily_visits.filter((pl.col("brand_happiness") <= 100) & (pl.col("brand_happiness") >= 1))
        avg_daily_brand_happiness = daily_visits.group_by("datetime").agg(pl.col("brand_happiness").mean())

        # Merge all the data together
        final_table = daily_advertisements.join(avg_daily_brand_happiness, on="datetime")
        final_table = final_table.join(self.revenue, on="datetime")
        return final_table
