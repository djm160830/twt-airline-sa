import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from prettytable import PrettyTable
 

# TODO 1 & 2
def read_tweets():
	return pd.read_csv("https://raw.githubusercontent.com/djm160830/twt-airline-sa/master/archive/Tweets.csv", 
		usecols=["airline_sentiment", "airline", "text"])

# TODO 3
def preprocess(data):
	# Convert text to lowercase 
	for column in data.columns:
		data[column] = data[column].str.lower()

	# Categorize target variable
	le = preprocessing.LabelEncoder()
	data['airline_sentiment'] = le.fit_transform(data['airline_sentiment'])

	# TODO 4: Split data into training and testing (10% testing) using train_test_split
	X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.10)
 
	# Transform training text using countvectorizer and tfidftransformer
	"""
	CountVectorizer(): Counts frequency of words
	TfidfTransformer(): Adjusts for the fact that some words appear more frequently in general (ex: 'we', 'the').
	"""
	count_vect = CountVectorizer() 
	counts = count_vect.fit_transform(X_train['text']) # Learn the vocabulary dictionary
	transformer = TfidfTransformer(use_idf=True)
	counts = transformer.fit_transform(counts) 			# Learn IDF vector (global term weights), and transform a count matrix to a tf-idf representation

	# Process test data
	X_test_cv = count_vect.transform(X_test['text']) 
	X_test_tf = transformer.transform(X_test_cv)

	# return data, counts, enc
	return counts, X_test_tf, y_train, y_test, data['airline_sentiment'], X_test

if __name__ == "__main__":
	t = PrettyTable(['ITERATION', 'ALPHA', 'FIT_PRIOR', 'TRAINING ACCURACY'])

	# TODO 7:  Repeat process 5 times with different parameter choices (for laplace smoothing & fit_prior) and output the parameters and accuracy in a tabular format.
	for i, a in enumerate(np.linspace(1.25, 1.0e-10, 5)):
	  for _, fp in enumerate([True, True, True, False, False, False]):
	    """
	    TODO 1: Read in tweets
	    TODO 2: Read in airline_sentiment, airline, and text into a dataframe
	    """
	    df = read_tweets()

	    """
	    TODO 3: Text preprocessing steps: 
	    - convert text to lowercase
	    - transform the training text using countvectorizer and tfidftransformer,
	    - convert airline_sentiment from categorical to numerical values using labelencoder
	    """
	    X_train, X_test, y_train, y_test, target_sentiment, X_test_raw = preprocess(df)

	    # TODO 5: Build a Multinomial Naïve Bayes (MNB) model using the training dataset
	    model = MultinomialNB(alpha=a, fit_prior=fp).fit(X_train, y_train)

	    # TODO 6: Apply model on test and output the accuracy
	    predicted = model.predict(X_test) 
	    accuracy = model.score(X_train, y_train)
	    t.add_row([i+1, a, fp, accuracy])
	  if i!=4: t.add_row([' ', ' ', ' ', ' '])	
	print(t)

	"""
	TODO 8: Outputting the average sentiment of each airline and
	reporting which airline has the highest positive sentiment.
	""" 
	df['airline_sentiment'] = target_sentiment
	highest_sentiment = df.groupby('airline').agg(mean_sentiment=('airline_sentiment', 'mean')).sort_values(by='mean_sentiment', ascending=False)
	print(f'\n{highest_sentiment}')
	print(f'\nHighest positive sentiment: \n{highest_sentiment[:1]}')

	# Model's prediction of US airline sentiments
	X = X_test_raw
	p = X["airline"].reset_index().join(pd.Series(predicted, name="sentiment"))
	print(f'\n{p.groupby("airline").agg(mean_sentiment=("sentiment", "mean")).sort_values(by="mean_sentiment", ascending=False)}')

	# Model accuracy
	print(f'\n{metrics.accuracy_score(y_test, predicted)}')
