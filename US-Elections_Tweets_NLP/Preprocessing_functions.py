# this is the file with useful functions in data preprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# custom lib
from NLP_functions import clean
from NLP_functions import get_the_lenguages
from sentiment_analysis import sentiment_analysis


# this dataframe parellelize the worload on pandas operations over the dataframe, credits to
# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def safe_drop_attr(df, list_drop):
    """
    This function is just a variation of the usual drop, we are using it just for clarity
    :param df: Pandas dataframe
    :param list_drop: list containing the labels to be dropped
    :return: the df without the given label
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    # drop the label contained in list_drop
    df.drop(columns=list_drop, inplace=True)

    return df


def safe_eliminate_NaN(df):
    """
    This function eliminates rows containing NaN values from the given DF. It also tells how many features were
    eliminated in that process
    :param df: pandas DataFrame
    :return: the df without NaN values
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a DF!")

    inst_before = df.shape[0]
    # drop rows containing NaN values
    df.dropna()
    inst_after = df.shape[0]
    print("The number of instances after dropped are: ", inst_before - inst_after)

    return df


def text_preprocessing(df):
    """
    This function preprocesses the data for the 2020 USA election dataset.
    1) clean the text of the tweets
    2) creates a column in the dataframe with the languages of each tweet
    3) perform the sentiment analysis
    4) deletes the text of the tweets
    :param df:
    :return:
    """
    tqdm.pandas(desc="Clean the data: ")
    # PREPROCESSING
    # create the text mined features
    df['clean_tweet'] = df['tweet'].progress_apply(clean)

    # find the list of the leanguages the tweets where written in
    df = get_the_lenguages(df)

    return df


def text_mining(df):
    # divide the data into states (via groupby) and get the state percentage of twitter english speakers!
    my_groupby = df.groupby(["STATE_NAME"])
    groups = dict(list(my_groupby))
    states_names = groups.keys()

    # empty list
    en_df = []
    total_speakers_df = []

    # get the number of tweets in english and the total number of tweets
    for name in states_names:
        temp = groups[name][groups[name]["Languages"] == 'en']
        # remember languages is a dict
        en_df.append(temp["Languages"].count())
        total_speakers_df.append(groups[name]["Languages"].count())

    # get the percentage
    en_df = np.array(en_df)
    total_speakers_df = np.array(total_speakers_df)
    perc_en = np.divide(en_df, total_speakers_df)

    # create a pandas dataframe
    share_df = pd.DataFrame({"STATE_NAME": states_names, "%_english": perc_en})

    # perform sentiment analysis
    df = parallelize_dataframe(df, sentiment_analysis, n_cores=3)

    # get rid of the text (we don't need them)
    df.drop(columns=['tweet', 'clean_tweet'], inplace=True)

    return df, share_df


def select_dates_tweets(df):
    """
    This function only works on the kaggle Twitter 2020 US dataset.
    It selects the dates before the last public debate, before the elections and after the election day.
    :param df: in our code it is the dataset after the sentiment analysis, in general it can also be applied to the
                starting kaggle dataset
    """
    # create a copy!
    df = df.copy()

    # create a slice of our df
    timestamps = df["created_at"]

    # decostruct the time variable
    df["month"] = [int(st[5:7]) for st in timestamps]
    df["day"] = [int(st[8:10]) for st in timestamps]

    # the data are only present in October and November
    # last debate
    df_last_debate = df.loc[(df['month'] == 10) & (df['day'] <= 22)].copy()

    # day after election day
    df_election_day = df.loc[((df['month'] == 11) & (df['day'] <= 3)) | (df['month'] == 10)].copy()

    # drop the month and day informations since they are integers and we do not need them in the next part!
    df.drop(columns=["month", "day"], inplace=True)
    df_last_debate.drop(columns=["month", "day"], inplace=True)
    df_election_day.drop(columns=["month", "day"], inplace=True)

    return df_last_debate, df_election_day, df

