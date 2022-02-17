# this lib contains NLP functions

# lib
import re

from spacy.language import Language
from spacy_langdetect import LanguageDetector
from tqdm import tqdm

import spacy
# boost computations
spacy.prefer_gpu()


# initializing the nlp pipeline
@Language.factory("language_detector")
# define the function
def get_lang_detector(nlp, name):
    """
    Proxy function for language detection.
    """
    return LanguageDetector()


# load english
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('language_detector', last=True)


# Sentiment analysis
# Language detection using spacy
# More details here: https://spacy.io/universe/project/spacy-langdetect
def clean(text):
    """
    This function takes a string and cleans it from anything that is not a character also it lowers all characters.
    :param text: the string of text that should be cleaned
    :return: the cleaned text
    """
    text = str(text).lower()
    text = re.sub("[^a-z]", ' ', str(text))
    return text


def get_the_lenguages(df, col_name="clean_tweet"):
    """
    take the cleaned tweets and create an empty list, then loops through tweets and add language to the list
    :param col_name: key to access the tweets in the dataframe
    :param df: dataframe we are working on
    :return: the list of the languages in the twitter database
    """
    # initialize tqdm
    tqdm.pandas()

    df["Languages"] = df[col_name].progress_apply(lambda x: nlp(x)._.language['language'])
    df["Languages"].dropna(inplace=True)

    return df
