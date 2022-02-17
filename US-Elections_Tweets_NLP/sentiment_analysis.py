'''
Set of functions for sentiment analysis, using TextBlob. This is only for English language.
We'll need to find a model trained  for Spanish in order to do proper sentiment analysis. 
'''
import matplotlib as mpl
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Plotting library with some statistical tools
from textblob import TextBlob
from tqdm.auto import tqdm

biden_color = '#2986cc'
trump_color = '#cc0000'


def getSubjectivity(text):
    """
    Function to get subjectivity of a string
    Input: Tweet to be analyzed
    Output: Subjectivity value
    """
    s = TextBlob(text)
    return s.sentiment.subjectivity


def getPolarity(text):
    """
    Function to get polarity of a string
    Input: Tweet to be analyzed
    Output: Polarity value
    """
    p = TextBlob(text)
    return p.sentiment.polarity


def getAnalysis(score):
    """
    Function tags tweets as negative, positive or neutral based on polarity value
    Input: Polarity Score
    Output: Negative, Neutral or Positive Tag.
    """
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def sentiment_analysis(dataset):
    """
     Function used to do sentiment analysis. Calculates Polarity value [-1,1] and subjectivity value [0,1].
     For polarity, negative values exhibit negative sentiment and vice versa. Zero value suggests neutrality.
     For subjectivity, 0 is completely objective statement and 1 is completely subjective statement.
     Input: The whole dataset to be subjected to sentiment analysis
     Output: Dataset with added features such as Polarity, Subjectivity and Overall Sentiment (positive/negative)
     """
    tqdm.pandas(desc="Sentiment analysis")
    # Run sentiment analysis on tweets and add polarity and subjectivity features
    dataset['TextBlob_Subjectivity'] = dataset['tweet'].progress_apply(getSubjectivity)
    dataset['TextBlob_Polarity'] = dataset['tweet'].progress_apply(getPolarity)
    # Add feature to dataset
    dataset['TextBlob_Analysis'] = dataset['TextBlob_Polarity'].progress_apply(getAnalysis)
    return dataset


def calc_stat_sentiment(dataset1, dataset2, key):
    """
    Function to calculate average polarity and subjectivity per state
    Input: The two datasets and a key statement for Subjectivity or Polarity
    Output: Lists with the average value of each state
    """
    biden_origin_state = dataset1.state.value_counts()[:]
    biden_avg = []
    trump_avg = []
    if key == 'Subjectivity':
        key_1 = 'TextBlob_Subjectivity'
    elif key == 'Polarity':
        key_1 = 'TextBlob_Polarity'
    else:
        print(key)
        print('Please check function use again')
    print('Caclulating average...')
    for i in biden_origin_state.index:
        print(i)
        temp_set = dataset1[dataset1.state == i]
        # temp_val=temp_set[key].sum()/temp_set.shape[0]
        biden_avg.append(temp_set[key_1].mean())
        temp_set = dataset2[dataset2.state == i]
        # temp_val=temp_set[key].sum()/temp_set.shape[0]
        trump_avg.append(temp_set[key_1].mean())
    return biden_avg, trump_avg


def plot_sentiment_overall(dataset1, dataset2, key):
    """
    Function to produce probality density function plots for subjectivity and polarity of the whole dataset
    Input: Dataset to be used, dataset should contain Polarity and Subjectivity features. Key is which feature should be used.
    Ouput: Plot requested by the key, stored in the folder on which the program is run
    """
    if key == 'Polarity':
        sns.kdeplot(data=dataset1, x='TextBlob_Polarity', label='Biden', color=biden_color)
        sns.kdeplot(data=dataset2, x='TextBlob_Polarity', label='Trump', color=trump_color)
        plt.title('Polarity Distribution Function', fontsize=15)
        plt.legend(frameon=False)
        plt.xlabel('Polarity')
        plt.savefig('total_polarity_distr.jpg', dpi=300, bbox_inches='tight')
        plt.clf()
    elif key == 'Subjectivity':
        sns.kdeplot(data=dataset1, x='TextBlob_Subjectivity', label='Biden', color=biden_color, linestyle='-')
        sns.kdeplot(data=dataset2, x='TextBlob_Subjectivity', label='Trump', color=trump_color, linestyle='-')
        plt.title('Subjectivity Distribution Function', fontsize=15)
        plt.legend(frameon=False)
        plt.xlabel('Subjectivity')
        plt.savefig('total_subj_distr.jpg', dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        print('Please read function use')


def plot_sentiment_state(dataset1, dataset2, key):
    """
    Function to produce probability density function plots for subjectivity and polarity of the whole dataset
    Input: Dataset to be used, dataset should contain Polarity and Subjectivity features. Key is which feature should be
    used.
    """
    biden_origin_state = dataset1.state.value_counts()[:]
    if key == 'Subjectivity':
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(17, 33)
        count = 0
        for i in range(13):
            for j in range(4):
                plt.subplot2grid((13, 4), (i, j))
                plt.title(biden_origin_state.index[count])
                plt.xlabel('Subjectivity')
                sns.kdeplot(data=dataset1[dataset1.state == biden_origin_state.index[count]], x='TextBlob_Subjectivity',
                            label='Biden', color=biden_color)
                sns.kdeplot(data=dataset2[dataset2.state == biden_origin_state.index[count]], x='TextBlob_Subjectivity',
                            label='Trump', color=trump_color)
                plt.legend(frameon=False, fontsize=12)
                count += 1
        fig.tight_layout()
        plt.savefig('subj_by_state.jpg', dpi=300, bbox_inches='tight')
        plt.clf()

        biden_avg_subj, trump_avg_subj = calc_stat_sentiment(dataset1, dataset2, 'Subjectivity')
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.yticks(fontsize=30)
        plt.bar(biden_origin_state.index, biden_avg_subj, color=biden_color, label='Biden', alpha=0.8)
        plt.bar(biden_origin_state.index, trump_avg_subj, color=trump_color, label='Trump', alpha=0.8)

        plt.title("Subjectivity by state")
        plt.legend(frameon=False, fontsize=30)
        plt.xticks(rotation=90)
        plt.ylabel('Subjectivity', fontsize=35)

        plt.savefig('avg_sub_state_en.jpg', dpi=300, bbox_inches='tight')
        plt.clf()
    elif key == 'Polarity':
        # Polarity Distribution per state ####
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(17, 33)
        count = 0
        for i in range(13):
            for j in range(4):
                plt.subplot2grid((13, 4), (i, j))
                plt.title(biden_origin_state.index[count])
                plt.xlabel('Polarity')
                sns.kdeplot(data=dataset1[dataset1.state == biden_origin_state.index[count]], x='TextBlob_Polarity',
                            label='Biden', color=biden_color)
                sns.kdeplot(data=dataset2[dataset2.state == biden_origin_state.index[count]], x='TextBlob_Polarity',
                            label='Trump', color=trump_color)
                plt.legend(frameon=False, fontsize=15)
                count += 1
        fig.tight_layout()
        plt.savefig('pol_by_state.jpg', dpi=300, bbox_inches='tight')
        plt.clf()
        biden_avg_pol, trump_avg_pol = calc_stat_sentiment(dataset1, dataset2, 'Polarity')
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.yticks(fontsize=30)
        plt.bar(biden_origin_state.index, biden_avg_pol, color=biden_color, label='Biden', alpha=0.8)
        plt.bar(biden_origin_state.index, trump_avg_pol, color=trump_color, label='Trump', alpha=0.8)
        plt.ylabel('Polarity', fontsize=35)
        plt.title("Polarity by state")
        plt.legend(frameon=False, fontsize=30)
        plt.xticks(rotation=90)
        plt.savefig('avg_pol_state.jpg', dpi=300, bbox_inches='tight')
        plt.clf()

    elif key == 'Correlation':
        biden_avg_pol, trump_avg_pol = calc_stat_sentiment(dataset1, dataset2, 'Polarity')
        biden_avg_subj, trump_avg_subj = calc_stat_sentiment(dataset1, dataset2, 'Subjectivity')
        plt.scatter(biden_avg_subj, biden_avg_pol, color=biden_color, label='Biden States')
        plt.scatter(trump_avg_subj, trump_avg_pol, color=trump_color, label='Trump States')
        plt.ylabel('Polarity')
        plt.xlabel('Subjectivity')
        plt.legend(frameon=False, loc=(0.6, 0.0))
        plt.savefig('pol_vs_subj_en.jpg', dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        print('Please read function use')
