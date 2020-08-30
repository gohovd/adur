import logging
import os
import pandas as pd
import numpy as np
import statistics as s
import re
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

PATH_LIST = os.path.abspath(__file__).split(os.sep)
ROOT_DIR = PATH_LIST[0:len(PATH_LIST)-2]
REVIEWS_DIR = "/".join(ROOT_DIR) + "/data/reviews/"
CLEANED_DIR = "/".join(ROOT_DIR) + "/data/cleaned/"

log = logging.getLogger(__name__)

class Cleaner:

	def __init__(self, reviews=None, year=None, min_words=None):
		'''
		This class is made to handle user reviews scraped from Google Play.

		Constructor requires a path to the reviews to kick off processing
		'''

		self.reviews_file = reviews
		self.year = year
		self.min_words = int(min_words)

		try:
			CLEANED_FILE_ABSOLUTE = CLEANED_DIR + reviews[:-4] + "_cleaned.csv"
			cleaned_dataframe = pd.read_csv(CLEANED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True).loc[year]
			log.info("Found cleaned version, skipping data-cleaning.")
			self.dataframe = cleaned_dataframe
			cleaned_version_exists = True
		except Exception as e:
			cleaned_version_exists = False
			log.warning(f'Could not find cleaned version, {str(e)}')

		if not cleaned_version_exists:
			REVIEW_FILE_ABSOLUTE = REVIEWS_DIR + reviews
			self.dataframe = pd.read_csv(REVIEW_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True).loc[year]
			self.clean()

		self.review_count = len(self.dataframe)
		self.earliest_review_date = self.dataframe.index.min()
		self.latest_review_date = self.dataframe.index.max()
		self.upvote_average = round(s.mean(self.dataframe.Rating), 2)
		self.upvote_ratio = round(sum(self.dataframe.Likes)/len(self.dataframe))
		self.upvote_variance = round(s.variance(self.dataframe.Likes), 2)
		self.upvote_max = max(self.dataframe.Likes)
		self.upvote_max_text = self.dataframe.Review[self.dataframe.Likes == max(self.dataframe.Likes)].values[0]
		self.upvote_top_five = self.dataframe.Likes.nlargest(5).values

		# self.print_sample()
		self.print_info()

	def clean(self):
		''' Executes the various functions to clean the dataset, exports clean .csv to root/data/cleaned '''
		self.add_time_columns()
		self.add_columns_for_review_length()
		self.convert_replydates()
		self.drop_null_reviews()
		self.drop_short_reviews()
		new_filename = self.reviews_file[:-4] + "_cleaned.csv"
		self.dataframe.to_csv(CLEANED_DIR + new_filename, columns = self.dataframe.columns)
		log.info(f'Cleaning complete. Saved cleaned dataset: {new_filename}')

	def print_sample(self):
		print('\n')
		print(self.dataframe.sample(5))
		print('\n')

	def print_info(self):
		print('Dataset information:\n')
		print(f"Reviews: {self.review_count}")
		print(f"First review: {self.earliest_review_date}")
		print(f"Latest review: {self.latest_review_date}")
		print(f"Average rating: {self.upvote_average}")
		print(f"Upvotes per review: {self.upvote_ratio}")
		print(f"Upvotes variance: {self.upvote_variance}")
		print(f"Most upvoted review: {self.upvote_max}")
		print(f"Five most upvoted: {self.upvote_top_five}")
		print(f'Most upvoted review text: "{self.upvote_max_text}"')
		print('\n')

	def add_time_columns(self):
		log.debug(f'Adding time based columns to dataframe..')
		start = time.time()
		if not 'Year' in self.dataframe:
			self.dataframe['Year'] = self.dataframe.index.year
		if not 'Month' in self.dataframe:
			self.dataframe['Month'] = self.dataframe.index.month
		if not 'Weekday Name' in self.dataframe:
			self.dataframe['Weekday Name'] = self.dataframe.index.weekday
			weekday_names = []
			for index, row in self.dataframe.iterrows():
				weekday_names.append(Cleaner.weekday_num_to_name(row['Weekday Name']))
			self.dataframe['Weekday Name'] = weekday_names
		log.debug(f'[{round((time.time() - start), 2)}s] Added/verified time-based columns [Year, Month, Weekday Name]')

	def weekday_num_to_name(weekday_num):
		num_to_day = {
			0 : 'Monday',
			1 : 'Tuesday',
			2 : 'Wednesday',
			3 : 'Thursday',
			4 : 'Friday',
			5 : 'Saturday',
			6 : 'Sunday'
		}
		return num_to_day[weekday_num]

	def drop_short_reviews(self):
		start, to_drop, old_length = time.time(), self.get_short_reviews("word", self.min_words), len(self.dataframe)
		if len(to_drop) > 0:
			self.dataframe = self.dataframe.drop(to_drop)
		log.debug(f'[{round((time.time() - start), 2)}s] Was: {old_length} Now: {len(self.dataframe)}. Dropped: {old_length - len(self.dataframe)} (-{round((old_length - len(self.dataframe))/(old_length / 100))}%)')
		

	def get_short_reviews(self, mode, minimum):
		log.debug('Removing short reviews, this may take a while..')
		indices_to_drop = []
		for index, row in self.dataframe.iterrows():
			review = row['Review']
			if isinstance(review, str):
				if mode == "word":
					words = re.split(" |-", review)
					if len(words) < minimum:
						indices_to_drop.append(index)
				else:
					if len(review) < minimum:
						indices_to_drop.append(index)
		return indices_to_drop

	def drop_null_reviews(self):
		dropped, start = 0, time.time()
		for index, row in self.dataframe.iterrows():
			if type(row['Review']) != str:
				dropped += 1
				self.dataframe = self.dataframe.drop(index)
		log.debug(f'[{round((time.time() - start), 2)}s] Dropped {dropped} null review(s)')

	def convert_replydates(self):
		''' Replace null values in ReplyDate column with 0, and datetime values with 1 '''
		replydates, start = [], time.time()
		for entry in self.dataframe.ReplyDate:
			if type(entry) != str:
				replydates.append(0)
			else:
				replydates.append(1)
		self.dataframe.ReplyDate = replydates
		log.debug(f'[{round((time.time() - start), 2)}s] ReplyDates converted/verified')

	def add_columns_for_review_length(self):
		start = time.time()
		if 'Words' in self.dataframe and 'Chars' in self.dataframe:
			log.debug('Review length columns already present in dataframe')
			return
		review_wordcount = []
		review_charcount = []
		for review in self.dataframe['Review'].astype('str'):
			review_wordcount.append(len(re.split(" |-", review)))
			review_charcount.append(len(review))
		self.dataframe.insert(3, 'Words', review_wordcount)
		self.dataframe.insert(4, 'Chars', review_charcount)
		log.debug(f'[{round((time.time() - start), 2)}s] Added/verified review-length columns [Chars, Words]')


	def vis(self):
		yaxis = self.dataframe['Rating'].loc['2019-12-31']
		xaxis = range(0,len(yaxis))
		
		fig = plt.figure(figsize=(10, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
		plt.style.use('fast')
		plt.xticks(np.arange(min(yaxis), max(yaxis)+1, 1))
		plt.plot(xaxis, yaxis, linewidth=1)
		plt.title("MY TITLE")
		plt.show()