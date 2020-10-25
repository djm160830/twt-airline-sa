# NAME:			Darla Maneja
# EMAIL:		djm160830@utdallas.edu
# SECTION:		CS4372.001
# Assignment 4

import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pdb


# TODO 1 & 2
def read_tweets():
	# File is already downloaded. Import it from its path.
	print(f'arguments: {list(enumerate(sys.argv))}')
	file = " ".join(sys.argv[1:])
	return pd.read_csv(file, usecols=["airline_sentiment", "airline", "text"])


# TODO 3
def preprocess(data):
	# Convert text to lowercase 
	for column in data.columns:
		data[column] = data[column].str.lower()

	# Transform text using countvectorizer and tfidftransformer
	"""
	CountVectorizer(): Counts frequency of words
	TfidfTransformer(): Adjusts for the fact that some words appear more frequently in general (ex: 'we', 'the').
	"""
	count_vect = CountVectorizer() 
	counts = count_vect.fit_transform(data['text']) # Learn the vocabulary dictionary
	transformer = TfidfTransformer().fit(counts) # Learn the IDF vector (global term weights).  
	counts = transformer.transform(counts) # Transform a count matrix to a tf or tf-idf representation

	# Convert airline_sentiment from categorical to numerical values using labelencoder
	le = preprocessing.LabelEncoder()
	enc = le.fit_transform(data['airline_sentiment'])
	# Maybe I put this in the train_test_split
	# data['airline_sentiment'] = enc
	return data, counts, enc

if __name__ == "__main__":
	# TODO 1: Read in tweets & TODO 2: Read in airline_sentiment, airline, and text into a dataframe
	df = read_tweets()

	"""TODO 3: Perform the following text preprocessing steps: 
	- convert text to lowercase, 
	- transform the text using countvectorizer and tfidftransformer,
	- convert airline_sentiment from categorical to numerical values using labelencoder
	"""
	df, counts, y_encoded = preprocess(df)

	# TODO 4: Split data into training and testing (10% testing) using train_test_split from scikit learn
	X_train, X_test, y_train, y_test = train_test_split(counts, y_encoded, test_size=0.1)

	# TODO 5: Build a Multinomial Naïve Bayes (MNB) model using the training dataset. You have to choose the best set of parameters.
	model = MultinomialNB().fit(X_train, y_train)
	predicted = model.predict(X_test)

	pdb.set_trace()


	# TODO 6: Apply your model on test and output the accuracy

	# TODO 7:  Repeat this process 5 times with different parameter choices and output the parameters and accuracy in a tabular format.

	"""TODO 8: Answer this question:
	The following is not related to naïve Bayes, but you can use the above data to answer the
	following question:
	Using the numeric value of airline_sentiment, output the average sentiment of each airline and
	report which airline has the highest positive sentiment.
	""" 
