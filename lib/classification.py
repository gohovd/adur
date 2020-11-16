import re
import numpy as np
import pandas as pd
import statistics as s
import warnings
import logging
import unicodedata
import inflect

from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

class Classifier:
	
	def __init__(self, reviews, labels):
		log.info(f'classifier initialized with file: {reviews} and labels {labels}')
		self.dataframe = pd.read_csv(reviews, index_col='Timestamp', parse_dates=True)
		self.reviews = self.dataframe.Review
		self.labels = pd.read_csv(labels, index_col='index')

		print(self.labels)

		# self.train_classifier(max_features=5000, random_state=42, train_size=0.7)

	def train_classifier(self, max_features=5000, random_state=8, train_size=0.7):
		''' Encode target labels with values between 0 and n_classes-1 '''
		encoder = LabelEncoder()
		encoded_labels = encoder.fit_transform(self.labels)

		''' Split training and test data (70/30) '''
		X_train, X_test, y_train, y_test = model_selection.train_test_split(
			self.reviews,
			encoded_labels,
			random_state=random_state,
			train_size=train_size
		)

		''' term frequency inverse document frequency '''
		tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
		tfidf_vectorizer.fit(reviews) # fit on the entire vocabulary

		''' Model(s) '''
		log_reg = LogisticRegression(solver='liblinear', multi_class='auto')
		log_reg = log_reg.fit(X_train_vect, y_train)
		log_reg_predictions = log_reg.predict(X_test_vect)

		''' Printing '''
		self.print_performance('logistic regression', y_test, log_reg_predictions)

	def print_performance(title, test, predicted):
		print(f'/\/\/\/\/ {title}')
		print("Recall\t\t",recall_score(test, predicted, average="weighted")*100)
		print()
		print("Precision\t", precision_score(test, predicted, average='weighted')*100)
		print()
		print("F1\t\t", f1_score(test, predicted, average='weighted')*100, "\n")

	def get_tfidf(self, reviews, X_train):
		tfidf_vectorizer.fit(reviews)
		X_train_tfidf = tfidf_vectorizer.transform(X_train)
		return X_train_tfidf

	def get_count_vect(self, X_train, X_test):
		count_vectorizer.fit(X_train)
		X_train_vect = count_vectorizer.transform(X_train)
		return X_train_vect

	def add_features(self, dataframe, features_list):
		enriched_reviews = []
		for index, row in dataframe.iterrows():

			features = []
			for nf in features_list:
				features.append(row[f])

			enriched_review = ""
			for f in features:
				enriched_review += str(feature) + " "
			enriched_review += row['review']
			enriched_reviews.append(enriched_review)
		
		return enriched_reviews

	def stringify_reviews(self, reviews):
		new_reviews = []
		for review in reviews:
			new_reviews.append(str(review))
		return new_reviews