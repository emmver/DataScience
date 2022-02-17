import matplotlib as mpl
import matplotlib.pyplot as plt  # Plotting library
import pandas as pd  # Library for data manipulation and analysis

from sentiment_analysis import *

mpl.rcParams['figure.dpi'] = 400
plt.rcParams["font.family"] = "Ubuntu"
plt.style.use('C:/Users/Toumba/Documents/plotstyle.mplstyle')
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['xtick.minor.visible']=True
# plt.rcParams['ytick.minor.visible']=True
# plt.rcParams['xtick.minor.size'] = 5
# plt.rcParams['ytick.minor.size'] = 5
# plt.rcParams['xtick.minor.width'] = 1.5
# plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['ytick.major.pad'] = '8'
#############################################

biden_color = '#2986cc'
trump_color = '#cc0000'

keys_plotting = ['Subjectivity', 'Polarity', 'Correlation']

temp = pd.read_csv('hashtags_trump_lang.csv')
temp = temp.drop(columns=['Unnamed: 0', 'TextBlob_Subjectivity', 'TextBlob_Polarity', 'TextBlob_Analysis'], axis=1)
trump_set_clean = temp[temp.languages == 'en'].copy()
temp = pd.read_csv('hashtags_biden_lang.csv')
temp = temp.drop(columns=['Unnamed: 0', 'TextBlob_Subjectivity', 'TextBlob_Polarity', 'TextBlob_Analysis'], axis=1)
biden_set_clean = temp[temp.languages == 'en'].copy()

print("Running Sentiment Analysis...")
trump_sentiment_analysis = sentiment_analysis(trump_set_clean)
biden_sentiment_analysis = sentiment_analysis(biden_set_clean)

print("Now Plotting")
plot_sentiment_overall(biden_sentiment_analysis, trump_sentiment_analysis, keys_plotting[0])
print('Plotting overall Polarity')
plot_sentiment_overall(biden_sentiment_analysis, trump_sentiment_analysis, keys_plotting[1])
print('Plotting Subjectivity by state')
plot_sentiment_state(biden_sentiment_analysis, trump_sentiment_analysis, keys_plotting[0])
print('Plotting Polarity by state')
plot_sentiment_state(biden_sentiment_analysis, trump_sentiment_analysis, keys_plotting[1])
print('Plotting Polarity vs Subjectivity for each state')
plot_sentiment_state(biden_sentiment_analysis, trump_sentiment_analysis, keys_plotting[2])
