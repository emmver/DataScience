# Data Science
Data science related projects - for fun


This project, as a title suggests, is an attempt by me to explore the world of Data Science. 
The projects here are mostly curiosity driven and are typically personal endeavors, not professional. 
Even though I have a few years of experience coding in Python, I'm quite new to the Data Science world, and I'm curious to learn more with these projects. 

As of now, there are four projects projects.

1) The US-Elections project. This project was part of a Data Science course I followed in Winter Semester 2021-2022 at the University of Vienna (Course: Doing Data Science). 
   It was part of the assesement for the course, and it was a group project, along with 3 excellent colleagues. The databases we were given consisted of approximately 2 million      tweets in the period from October 2020 to first week of January 2021 ( https://www.kaggle.com/manchunhui/us-election-2020-tweets ). After EDA (code provided) we tried to split    the dataset based on three crucial dates (Last debate, Election Day, Results announcements). Our analysis is based on Geolocalization (Geopandas), Sentiment analysis of each      state (TextBlob and Spacy) and clustering (scikit-learn) based on PCA of the sentiment analysis and other qualities such as Likes, Retweet counts etc. Eventually, our goal is      to build a tool that can recognize swing states, at least from a social media standpoint, where political campaigns can invest more. 
   
2) A COVID-19 visualization tool based on Python using the database from Our World In Data. 

3) Prediction of the income of a person based on years of education and years of working experience.The database used is from the National Longitudinal Survey of Youth 1997-2011. This database is one typically used by US social scientists. It contains data about the earnings, educational status, ethnicity, financial status of the family when the subject was growing up and several other features. 
   The approach used here is to split the set into a train and test dataset, and perform multiple linear regression for the earnings based on the years of education and experience.
   Judging from the outcome, the model can give approximate predictions however the error is still large. This could be improved by modelling the data with a Decision Tree Regression or maybe by introducing more features into the Linear Regression model.
