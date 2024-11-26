import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import os
from time_function import timer

# import pyspark.sql.functions as F
# from pyspark.sql import SparkSession


class PysparkMultiNode:
    """Class used to test pandas version 1"""

    def __init__(self, scenario: int):
        """Initate the class used to test pandas version 1"""
        self.scenario = scenario
        self.save_location = Path(os.getcwd()) / "data" / "data_files" / f"scenario_{scenario}"
        self.spark = SparkSession.builder.appName("spark_multi_node").getOrCreate()

    @timer
    def load_data(self):
        """TASK 0: Load the data from the save location"""
        customer_database = self.spark.read.parquet(self.save_location / f"customer_database.parquet")
        customer_database = customer_database.withColumn("customer_id", F.col("customer_id").cast("string"))
        customer_database = customer_database.withColumn("brand_happiness", F.col("brand_happiness").cast("float"))
        self.customer_database = customer_database

        store_visits = self.spark.read.parquet(self.save_location / f"store_visits.parquet")
        store_visits = store_visits.withColumn("customer_id", F.col("customer_id").cast("string"))
        store_visits = store_visits.withColumn("datetime", F.to_timestamp(F.col("datetime")))
        self.store_visits = store_visits

        advertisements = self.spark.read.parquet(self.save_location / f"advertisements.parquet")
        advertisements = advertisements.withColumn("datetime", F.to_timestamp(F.col("datetime")))
        advertisements = advertisements.withColumn("channel", F.col("channel").cast("string"))
        advertisements = advertisements.withColumn("spend", F.col("spend").cast("float"))
        self.advertisements = advertisements

        revenue = self.spark.read.parquet(self.save_location / f"revenue.parquet")
        revenue = revenue.withColumn("datetime", F.to_timestamp(F.col("datetime")))
        revenue = revenue.withColumn("revenue", F.col("revenue").cast("float"))
        self.revenue = revenue

    @timer
    def data_exploration(self):
        """TASK 1: Data exploration"""
        # Count the number of NaN values in brand_happiness
        num_nans = self.customer_database.filter(F.col("brand_happiness").isNaN()).count()
        print(f"Number of NaN values in brand_happiness: {num_nans}")

        # Display the top 10 customers by visit count
        visit_counts = self.store_visits.groupBy("customer_id").count().orderBy(F.col("count").desc()).limit(10)
        print(visit_counts)

        # Group by channel and calculate the average and sum of spend
        average_spend_per_channel = self.advertisements.groupBy("channel").agg(F.avg("spend").alias("average_spend"))
        sum_spend_per_channel = self.advertisements.groupBy("channel").agg(F.sum("spend").alias("sum_spend"))

        print(average_spend_per_channel)
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

        pandas_df = final_table.select(["brand_happiness", "mixed", "offline", "online", "revenue"]).toPandas()

        X = pandas_df[["brand_happiness", "mixed", "offline", "online"]]
        y = pandas_df["revenue"]

        X = sm.add_constant(X)
        statsmodels_model = sm.OLS(y, X).fit()
        print(statsmodels_model.summary())

    @timer
    def train_sklearn_model(self):
        """TASK 4: Train a simple linear regression model using Scikit-learn"""
        final_table = self._get_final_table()

        pandas_df = final_table.select(["brand_happiness", "mixed", "offline", "online", "revenue"]).toPandas()

        X = pandas_df[["brand_happiness", "mixed", "offline", "online"]]
        y = pandas_df["revenue"]

        # Initialize and fit the linear regression model
        statsmodels_model = LinearRegression()
        statsmodels_model.fit(X, y)

        # Print the coefficients and intercept
        print("Coefficients:", statsmodels_model.coef_)
        print("Intercept:", statsmodels_model.intercept_)

        # Print the R squared value
        y_pred = statsmodels_model.predict(X)
        print("R-squared:", r2_score(y, y_pred))

    def _get_final_table(self) -> pyspark.sql.DataFrame:
        """Create the final table"""
        # Aggregate per channel and pivot so that each channel is a column
        daily_advertisements = self.advertisements.groupBy("datetime", "channel").agg(F.sum("spend").alias("spend_sum"))
        daily_advertisements = daily_advertisements.groupBy("datetime").pivot("channel").sum("spend_sum")

        # Match the customers with their brand happiness and do some data cleaning
        daily_visits = self.store_visits.join(self.customer_database, "customer_id")
        daily_visits = daily_visits.filter(~F.col("brand_happiness").isNaN())
        daily_visits = daily_visits.filter((F.col("brand_happiness") <= 100) & (F.col("brand_happiness") >= 1))

        avg_daily_brand_happiness = daily_visits.groupBy("datetime").agg(
            F.avg("brand_happiness").alias("brand_happiness")
        )

        # Merge all the data together
        final_table = daily_advertisements.join(avg_daily_brand_happiness, "datetime", "inner")
        final_table = final_table.join(self.revenue, "datetime", "inner")

        return final_table
