# functions that are related to geography

import geopandas
import matplotlib.pyplot as plt
import pandas as pd


def drop_non_geolocalised(df, label_latitude, label_longitude):
    """
    This function selects only the elements of the df that are geolocalised
    :param df: dataframe with geolocalised data
    :param label_latitude: feature name of the latitude
    :param label_longitude: feature name of the longitude
    :return: the df with only geolocalized data
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    num_df = df.shape[0]
    print("The number of instances in the df is: ", num_df)

    # we remove the data we are not able to localize!
    # first we drop latitude
    df = df[df[label_latitude].notna()]

    # then we drop longitude
    df = df[df[label_longitude].notna()]

    num_df = df.shape[0]
    print("The number of instances after dropping the non localized records is: ", num_df)

    return df


def localize_tweets(df, title, plot=False):
    """
    Function that creates a shapely like object to be use to gelocalize the single tweets using geopandas
    :param plot: if true plots the data distribution over the world map
    :param df: Our dataframe with coordinates (lat and long). It plots te results on the world map!
    :param title: title of the plot
    :return: the geopandas dataframe with the geolocalized data!
    """
    # creating the geopandas dataset
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.long, df.lat))

    # importing world map
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    if plot:
        # plot the world map
        ax = world.plot(color='white', edgecolor='black')
        ax.set_title(title)

        # plot our points
        gdf.plot(ax=ax, marker=',', markersize=1, cmap='BuGn')

        plt.show()

    return gdf


def localize_USA(gdf, title, plot=False):
    """
    Function that creates a shapely like object to be use to gelocalize the single tweets using geopandas
    :param gdf: Our Geopandas dataframe with coordinates (lat and long). It plots te results in a USA map!
    :param title: title of the plot
    :return: the geopandas dataframe with the localized data!
    """
    # upload the USA map
    usa = geopandas.read_file("Maps/states.shp")
    gdf = geopandas.sjoin(usa, gdf, how="inner", op='contains')

    useless_col = ["country", "city", "state", "state_code", "lat", "long"]
    gdf.drop(columns=useless_col, inplace=True)

    if plot:
        # plot the world map
        ax = world.plot(color='white', edgecolor='black')
        ax.set_title(title)

        # plot our points
        gdf.plot(ax=ax, marker=',', markersize=1, cmap='BuGn')

        plt.show()

    print("The number of tweets available is: {} \n".format(gdf.shape[0]))

    return gdf


def assign_color(cluster_num):
    """
    This fucntion assigns the colours to the given elements of a clustering
    :param cluster_num: number used to identify the belongings to a particular cluster
    :return: a colour code
    """
    if cluster_num == 1:
        return 'lightsteelblue'
    if cluster_num == 2:
        return 'mediumblue'
    if cluster_num == 3:
        return 'cornflowerblue'


def plot_clustering_results(gdf):
    """
    This function plots the clustering results onto the USA map
    :param gdf: geopandas dataframe which includes a column "cluster_id" representing the clustering
    """
    gdf["Colour"] = gdf['cluster_id'].apply(assign_color)
    # adding a column with color names (gdf has 5 rows)
    gdf.plot(color=gdf['Colour'], figsize=(10, 15))
