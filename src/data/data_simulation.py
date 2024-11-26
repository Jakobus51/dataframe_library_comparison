from uuid import uuid4
import numpy as np
from datetime import datetime


class DataGenerator:
    def __init__(
        self,
        number_of_minutes: int,
        shop_visits_per_minute: int,
        advertisements_per_minute: int,
        customer_database_size: int,
    ):
        self.number_of_minutes = number_of_minutes
        self.shop_visits_per_minute = shop_visits_per_minute
        self.advertisements_per_minute = advertisements_per_minute
        self.customer_database_size = customer_database_size

        np.random.seed(42)
        START_TIME = np.datetime64("2021-01-01")
        self.minutes = np.arange(
            START_TIME,
            START_TIME + np.timedelta64(number_of_minutes, "m"),
            dtype="datetime64[m]",
        ).astype(str)

    def generate_and_set_data(self):
        now = datetime.now()
        self.customer_database = self._generate_customer_database()
        print(f"Generated customer database in {datetime.now() - now}")
        now = datetime.now()
        self.store_visits = self._generate_store_visits()
        print(f"Generated store visits in {datetime.now() - now}")
        now = datetime.now()
        self.advertisements = self._generate_advertisements()
        print(f"Generated advertisements in {datetime.now() - now}")
        self.revenue = self._generate_revenue()
        print(f"Generated revenue in {datetime.now() - now}")

    def _generate_customer_database(self) -> np.ndarray:
        """We simulate the customer database where there are some data errors:
        - 10% of the values are NaN
        - The scores are between -10 and 110, while only 0 to 100 is allowed

        Columns:
            - customer_id
            - brand_happiness
        """
        NAN_PERCENTAGE = 0.1
        BOTTOM_SCORE = -10
        TOP_SCORE = 110

        # Generate UUIDs
        customer_ids = np.fromiter(
            (uuid4().hex for _ in range(self.customer_database_size)), dtype="<U32", count=self.customer_database_size
        )
        # customer_ids = np.array([uuid4().hex for _ in range(self.customer_database_size)])

        brand_happiness = np.random.randint(BOTTOM_SCORE, TOP_SCORE, size=self.customer_database_size).astype(float)

        # Set 10% of the values to None
        none_indices = np.random.choice(
            self.customer_database_size,
            size=int(self.customer_database_size * NAN_PERCENTAGE),
        )
        brand_happiness[none_indices] = None
        customer_database = np.column_stack((customer_ids, brand_happiness))
        return customer_database

    def _generate_store_visits(self) -> np.ndarray:
        """We simulate the store visits where each day has a fixed amount of visits but by random people from the customer database

        Columns:
            - datetime
            - customer_id
        """
        store_visits = np.empty((self.shop_visits_per_minute * self.number_of_minutes, 2), dtype=object)

        customer_ids = self.customer_database[:, 0]
        # Populate each day with x random customer_ids
        for i, minute in enumerate(self.minutes):

            start_idx = i * self.shop_visits_per_minute
            end_idx = start_idx + self.shop_visits_per_minute

            # Assign date to the first column
            store_visits[start_idx:end_idx, 0] = minute

            # Assign random customer_ids to the second column
            store_visits[start_idx:end_idx, 1] = np.random.choice(customer_ids, self.shop_visits_per_minute)

        return store_visits

    def _generate_advertisements(self) -> np.ndarray:
        """We simulate the advertisements where each day has a fixed amount of advertisements

        Columns:
            - datetime
            - channel
            - spend
        """
        advertisements = np.empty((self.advertisements_per_minute * self.number_of_minutes, 3), dtype=object)

        # Populate each minute with x random advertisements
        for i, minute in enumerate(self.minutes):
            start_idx = i * self.advertisements_per_minute
            end_idx = start_idx + self.advertisements_per_minute

            # Assign datetime to the first column for the slice
            advertisements[start_idx:end_idx, 0] = minute

            # Assign a random channel to the second column for the slice
            advertisements[start_idx:end_idx, 1] = np.random.choice(
                ["offline", "online", "mixed"], self.advertisements_per_minute
            )

            # Assign random spend values to the third column for the slice
            advertisements[start_idx:end_idx, 2] = np.random.normal(10, 3, self.advertisements_per_minute)

        return advertisements

    def _generate_revenue(self) -> np.ndarray:
        """We simulate the revenue where each day has a fixed amount of revenue

        Columns:
            - datetime
            - revenue
        """
        revenue = np.empty((self.number_of_minutes, 2), dtype=object)

        # Populate each day with x random revenue
        for i, minute in enumerate(self.minutes):
            revenue[i, 0] = minute
            revenue[i, 1] = np.random.normal(
                100 * self.shop_visits_per_minute + 10 * self.advertisements_per_minute,
                (self.advertisements_per_minute + self.shop_visits_per_minute) * 10,
            )

        return revenue
