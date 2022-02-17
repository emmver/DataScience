# this is the main of the project

# libraries
import pandas as pd

# custom libraries
from Geographycal_functions import drop_non_geolocalised
from Geographycal_functions import localize_tweets
from Preprocessing_functions import parallelize_dataframe
from Preprocessing_functions import preprocessing

# import the raw data
data_donald = pd.read_csv("hashtag_donaldtrump.csv", lineterminator='\n')
data_joe = pd.read_csv("hashtag_joebiden.csv", lineterminator='\n')

# PREPROCESSING
data_donald = parallelize_dataframe(data_donald, preprocessing, n_cores=3)
data_joe = parallelize_dataframe(data_joe, preprocessing, n_cores=3)

# custom geololization
# data to drop
to_be_delete = ["tweet_id", "source", "user_id", "user_join_date", "user_location",
                "city", "country", "continent", "state", "state_code", "collected_at"]

# geolocalize Trump
print("Donald Trump")
data_donald = drop_non_geolocalised(data_donald, "lat", "long")

# geolocalize Joe
print("Joe Biden")
data_joe = drop_non_geolocalised(data_joe, "lat", "long")

# plotting Trump
geo_donald = localize_tweets(data_donald, "World Trump data distribution")

# plotting Biden
geo_biden = localize_tweets(data_donald, "World Joe data distribution")

# group the data by state
geo_donald = geo_donald.groupby("State").mean(numeric_only=True)
geo_biden = geo_biden.groupby("State").mean(numeric_only=True)
