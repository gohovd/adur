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
	'''
	This class produces an additional two datasets:
		1. Normalized (ready for classification) dataset, from the cleaned dataset
		2. Exportable (ready for anomaly detection by tsadf)

	Some columns may be more valuable with respect to classifier accuracy than others,
	it should be easy to plug-and-play another column, for experiment.
	'''

	def __init__(self, reviews=None, column=None, year=None, stem=None, lemmatize=None):
		log.info('initialized preprocessor')

		if year != None:
			self.year = year
		else:
			log.error('preprocesser needs the year to be specified')
			exit(0)

		CLEANED_FILE_ABSOLUTE = CLEANED_DIR + reviews[:-4] + "_cleaned.csv"
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

			exporter = TSADFExporter(self.dataframe, self.year, reviews)
			normalizer = Normalizer(self.dataframe, self.year, reviews, stem, lemmatize, column)
		except Exception as e:
			log.debug(e)
			exit(0)

class TSADFExporter:
	''' Prepares dataset for tsadf '''
	def __init__(self, dataframe, year, reviews):
		log.info('initialized tsadf exporter')

		try:
			EXPORTED_FILE_ABSOLUTE = EXPORTABLE_DIR + reviews[:-4] + "_exported.csv"
			self.dataframe = pd.read_csv(EXPORTED_FILE_ABSOLUTE, index_col='Time', parse_dates=True).loc[year]
			tsadf_version_exists = True
		except Exception as e:
			tsadf_version_exists = False
			log.error(str(e))

		if tsadf_version_exists:
			log.info('tsadf dataset already exists, skipping TSADFExporter..')
			return

		self.dataframe = dataframe
		self.monthly_dataframes = {}

		start = time.time()
		for i in range(1, 13):
			monthly = {}
			monthly[i] = self.make_monthly_dataframe(year=2019, month=i)
			
			df = pd.DataFrame.from_dict(monthly[i])
			df.index = pd.to_datetime(df['Day'])
			df = df.drop('Day', axis=1)
			self.monthly_dataframes[i] = df
		
		self.exportable = self.get_exportable_df(self.monthly_dataframes)

		new_filename = reviews[:-4] + "_exported.csv"
		self.exportable.to_csv(EXPORTABLE_DIR + new_filename, columns = self.exportable.columns)
		log.info(f'completed in {round((time.time() - start),2)}s, exporting to {new_filename}')

	def get_exportable_df(self, monthly_dataframes):
		'''
		Takes a dictionary holding one dataframe per month of the year.
		For each month, it calls the methods apped_zero_minutes_and_seconds
		which adds '00:00' to the datetime index col (this is necessary for 
		the tsadf library).

		Returns the concatenated version of the monthly dataframes, with the
		index column renamed to 'Time', with 00:00 suffixed each index.
		'''
		log.debug('building the last parts of exportable..')
		c = {
			'index' : [],
			'ReviewCount': [],
			'Words': [],
			'Chars': [],
			'Likes': [],
			'Rating': [],
			'Avg_Words': [],
			'Avg_Chars': [],
			'Avg_Likes': [],
			'Avg_Rating': []
    	}
		for i in range(1,13):
			new_index = self.append_zero_minutes_and_seconds(monthly_dataframes[i].index)
			c['index'] += new_index
			
			for col in monthly_dataframes[i].columns:
				c[col] += list(monthly_dataframes[i][col])
				
		exportable = pd.DataFrame.from_dict(c)
		exportable = exportable.set_index('index')
		exportable.index = pd.to_datetime(exportable.index, format="%Y-%m-%d %H:%S")
		exportable.index = exportable.index.strftime('%d.%m.%Y %H:%S')
		exportable.index.name = 'Time'
		return exportable

	def append_zero_minutes_and_seconds(self, indices):
		''' Appends 00:00 to a list of indices '''
		new_index = [];
		for index in indices:
			index = str(index)
			date = index.split(" ")[0]
			hours = index.split(" ")[1].split(":")[0]
			minutes = index.split(" ")[1].split(":")[1]
			index = date + " 00:00"
			new_index.append(index)
		return new_index

	def make_monthly_dataframe(self, year=2019, month=1):
		'''
		Takes a dataframe and a year, makes use of the monthrange library to
		find how many days per month in that year. For each day of each month,
		it sums and averages the values of each respective column -- on that day
		-- and puts in in a new dictionary of lists called 'data'.
		'''
		mrange = range(monthrange(year, month)[1])
		year_month = str(year) + "-" + str(month)
		month_dataframe_segment = self.dataframe.loc[year_month]
		df = self.dataframe

		data = {
			"Day": [],
			"ReviewCount": [],
			"Words": [],
			"Chars": [],
			"Likes": [],
			"Rating": [],
			"Avg_Words": [],
			"Avg_Chars" : [],
			"Avg_Likes": [],
			"Avg_Rating": []
		}
		
		for n in mrange:
			# Sum the values for each column for each day
			day = n + 1
			date = year_month + "-" + str(day)
			data["Day"].append(date);
			data["ReviewCount"].append(len(df.loc[date]))
			data["Words"].append(sum(df.loc[date]["Words"]))
			data["Chars"].append(sum(df.loc[date]["Chars"]))
			data["Likes"].append(sum(df.loc[date]["Likes"]))
			data["Rating"].append(sum(df.loc[date]["Rating"]))
			
			# Take the average of values for each column for each day
			if (len(df.loc[date]["Words"]) > 0):
				data["Avg_Words"].append(round(s.mean(df.loc[date]["Words"]), 2))
			else:
				data["Avg_Words"].append(0)
				
			if (len(df.loc[date]["Chars"]) > 0):
				data["Avg_Chars"].append(round(s.mean(df.loc[date]["Chars"]), 2))
			else:
				data["Avg_Chars"].append(0)
				
			if (len(df.loc[date]["Likes"]) > 0):
				data["Avg_Likes"].append(round(s.mean(df.loc[date]["Likes"]), 2))
			else:
				data["Avg_Likes"].append(0)
				
			if (len(df.loc[date]["Rating"]) > 0):
				data["Avg_Rating"].append(round(s.mean(df.loc[date]["Rating"]), 2))
			else:
				data["Avg_Rating"].append(0)
		
		# Finally return 
		return data

class Normalizer:
	''' Prepares dataset for text classification '''
	def __init__(self, dataframe=None, year=None, reviews=None, stem=True, lemmatize=False, column=None):
		log.info(f'initialized normalizer with settings - stem {stem} lemmatize {lemmatize} column {column} year {year}')

		self.reviews_file = reviews

		try:
			NORMALIZED_FILE_ABSOLUTE = NORMALIZED_DIR + reviews[:-4] + "_normalized.csv"
			self.dataframe = pd.read_csv(NORMALIZED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True).loc[year]
			log.info("Found normalized version, skipping normalization..")
			normalized_version_exists = True
			return
		except Exception as e:
			self.dataframe = dataframe
			normalized_version_exists = False
			log.warning(f'Could not find normalized version, {str(e)}')

		start = time.time()
		log.info('starting normalization of data...')

		if not normalized_version_exists:
			self.normalize(stem, lemmatize, column)

		log.info(f'normalization complete in {round(time.time() - start, 2)}s')
		self.dataframe.head(20)

	def normalize(self, stem, lemmatize, column):
		if stem == lemmatize:
			log.error('Please specify either stemming or lemmatization, not both..')
			exit(0)

		self.column = column		
		self.dataframe[self.column] = self.lowercase()
		self.dataframe[self.column] = self.remove_punctuation()
		self.dataframe[self.column] = self.asciionly()
		self.dataframe[self.column] = self.new_line_to_space()
		self.dataframe[self.column] = self.remove_single_characters()
		self.dataframe[self.column] = self.remove_stopwords()
		self.dataframe[self.column] = self.tokenize_words()
		if stem:
			self.dataframe[self.column] = self.stem()
		elif lemmatize:
			self.dataframe[self.column] = self.lemmatize()

		new_filename = self.reviews_file[:-4] + "_normalized.csv"
		self.dataframe.to_csv(NORMALIZED_DIR + new_filename, columns = self.dataframe.columns)

	def remove_punctuation(self):
		"""
		Uses string.punctuation list to remove unwarranted characeters
		!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
		"""
		punctuation_table = str.maketrans('', '', string.punctuation)
		stripped = [review.translate(punctuation_table) for review in self.dataframe[self.column]]
		return stripped

	def lowercase(self):
		review_arr = self.dataframe[self.column].to_list()
		return [review.lower() for review in review_arr]

	def asciionly(self):
		new_reviews = []
		for review in self.dataframe[self.column]:
			review = re.sub('[^A-Za-z0-9\s]', '', review)
			new_reviews.append(review)
		return new_reviews

	def new_line_to_space(self):
		new_reviews = []
		for review in self.dataframe[self.column]:
			nr = review.replace('\n', ' ')
			nr = nr.replace('\t', '') # remove tab as well
			new_reviews.append(nr)
		return new_reviews

	def remove_single_characters(self):
		new_reviews = []
		for review in self.dataframe[self.column]:
			words = review.split(" ")
			new_words = []
			for index, word in enumerate(words):
				if (len(word) > 1):
					new_words.append(word)
			new_review = " ".join(word for word in new_words)
			new_reviews.append(new_review)
		return new_reviews

	def remove_stopwords(self):
		extra_stop_words = {'doing','having','because','into','against','over','under','why','no','not','only','same','just','should','now'}
		stop_words = set(stopwords.words('english'))
		new_reviews = []
		for review in self.dataframe[self.column]:
			new_words = []
			for word in review.split(" "):
				if word not in stop_words:
					new_words.append(word)
			new_review = " ".join(w for w in new_words)
			new_reviews.append(new_review)  
		return new_reviews

	def tokenize_words(self):
		tokenized_reviews = []
		for review in self.dataframe[self.column]:
			tokenized_reviews.append(word_tokenize(review))
		return tokenized_reviews

	def stem(self):
		porter = PorterStemmer()
		new_reviews = []
		for review_tokens in self.dataframe[self.column]:
			stemmed = [porter.stem(word) for word in review_tokens]
			new_reviews.append(stemmed)
		return new_reviews

	def lemmatize(self):
		wnt = WordNetLemmatizer()
		new_reviews = []
		for review_tokens in self.dataframe[self.column]:
			lemmas = []   
			for word in review_tokens:
				lemma = wnt.lemmatize(word, pos='v')
				lemmas.append(lemma)
			new_reviews.append(lemmas)
		return new_reviews