import logging
import time
import string
import re
import os
import pandas as pd
import statistics as s
import numpy as np
from calendar import monthrange

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from lib.normalizer import Normalizer
from lib.tsadf_exporter import TSADFExporter

PATH_LIST 			= os.path.abspath(__file__).split(os.sep)
ROOT_DIR 			= PATH_LIST[0:len(PATH_LIST)-2]
REVIEWS_DIR 		= "/".join(ROOT_DIR) + "/data/reviews/"
CLEANED_DIR 		= "/".join(ROOT_DIR) + "/data/cleaned/"
PREPROCESSED_DIR 	= "/".join(ROOT_DIR) + "/data/preprocessed/"
EXPORTABLE_DIR	 	= "/".join(ROOT_DIR) + "/data/exportable/"

NORMALIZED_DIR 		= "/".join(ROOT_DIR) + "/data/normalized/"
TRAINING_DIR 		= "/".join(ROOT_DIR) + "/data/training/"

log = logging.getLogger(__name__)

class Preprocessor:
	'''Picks up where the Cleaner class left off; this class performs
	various pre-processing tasks on the dataframe in question, before
	passing it on to the next step.

	Keyword arguments:
	reviews - the absolute path to the dataframe
	column - the column in the dataframe holding the reviews
	year - the slice of time to preprocess, in the given dataframe
	stem -
	lemmatize -
	'''

	def __init__(self, reviews=None, column=None, year=None, stem=None, lemmatize=None):
		log.info(f'Preprocessor initialising with reviews: {reviews}, column: {column}, year: {year}, stem/lemma {stem}/{lemmatize}')
		# Year (i.e. time-slice) must be provided
		if year != None:
			self.year = year
		else:
			log.error('preprocesser needs the year to be specified')
			exit(0)

		# Assume there's a clean dataframe present..
		CLEANED_FILE_ABSOLUTE = CLEANED_DIR + reviews[:-4] + "_cleaned.csv"
		log.info("Absolute path of the cleaned dataset: %s" % CLEANED_FILE_ABSOLUTE)

		try:
			self.dataframe = pd.read_csv(CLEANED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True)

			if column == None:
				if 'Review' in self.dataframe.columns:
					self.column = 'Review'
				elif 'review' in self.dataframe.columns:
					self.column = 'review'
				else:
					logger.error('Could not find "Review" column, please specify to the preprocessor which column contains the user reviews.')
					exit(0)

			# everything happens in the __init__ function
			# only thing we have to do is make an instance
			tadf = TSADFExporter(reviews)

			# Call the Normalizer class to generate a dataset primed for classification
			# normalizer = Normalizer(self.dataframe, self.year, reviews, stem, lemmatize, column)
		except Exception as e:
			log.debug(e)
			exit(0)
