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
	
	def __init__(self):
		log.info('initialized classifier')

	def train_classifier(self, reviews, labels, max_features=5000):
		pass

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
		for index, row in dataframe.iterrows();

			features = []
			for nf in features_list:
				features.append(row[f])

			enriched_review = ""
			for f in features:
				enriched_review += str(feature) + " "
			enriched_review += row['review']
			enriched_reviews.append(enriched_review)
		
		return enriched_reviews

