import logging
import os
import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import statistics as s

PATH_LIST = os.path.abspath(__file__).split(os.sep)
ROOT_DIR = PATH_LIST[0:len(PATH_LIST)-2]

CLEANED_DIR = "/".join(ROOT_DIR) + "/data/cleaned/"
COLLAPSED_DIR = "/".join(ROOT_DIR) + "/data/collapsed/"
EXPORTED_DIR = "/".join(ROOT_DIR) + "/data/exportable/"
REVIEWS_DIR = "/".join(ROOT_DIR) + "/data/reviews/"

class Sampler:
	""" This is a stepping-stone program meant to produce a collection
	of review samples which will facilitate the assembly of a "gold
	standard" (or just measure), by which the parent program (Alert
	Monitoring Tool) will be evaluated.
		The goal of this class is to produce an excel document (.xlsx)
	with representative samples from a given dataset of user reviews
	(Dataset A).

	With the samples in place, the reviews and their attributes for
	each day are manually inspected, and each day is labelled either
	as an Alert, or as a non-Alert.

	To ensure a representative sample, reviews are taken from four
	distinct subsets of the user reviews dataset:
	peaks - the highest values
	valleys - the lowest values
	random -
	tadf - from time-series-anomaly-detection-framework
	A total of 4000 reviews are sampled from 40 days; a thousand
	from each of the respective subsets.
	
	To find out which subset a review belongs to, we produce -- based
	on the original dataset -- a new dataset (Dataset B), where all
	reviews belonging to the same day are collapsed into one single
	row, containing the sum totals and/or averages of the values for
	each column.

	It is on Dataset B that we employ statistical methods to discern 
	which review belongs to which of the above-mentioned subsets. First
	we choose which column we think is most relevant when it comes to
	the problem of determining whether the reviews written warrants the
	attention of the application's developer. For the purpose of this
	project, we opted for combining the two features 'review count' and
	'upvotes', into a new feature dubbed 'interactions' (count * upvotes).
		Having chosen a column, we find the peak values simply by
	collecting the indices of the highest values. Valleys we find by
	collecting the indices of the lowest values. To get ten indices from
	TADF, we first have to run TADF on Dataset B, targeting the column
	of interest, this will detect anomalous values and return their
	indices (printed to sysout). Lastly we pick ten random indices.
	
	A) Get a dataset of user reviews

	For a given dataset of reviews (say, 100.000 reviews scraped from
	the review section of a Google Play app) collected over a particular
	stretch of time (say, the year 2019) with a structure similar to the
	following:

	+---------------------+----------+--------------------------------+--------+---------+---------+
	| Timestamp  		  | Username | Review						  | Words  | Rating  | Upvotes |
	+---------------------+----------+--------------------------------+--------+---------+---------+
	| 08-08-2019 14:56:20 | George   | This app was fantastic, thanks |      7 |     5/5 |      27 |
	|            		  |          | to <particular function>       |        |         |         |
	+---------------------+----------+--------------------------------+--------+---------+---------+
	| 12-12-2019 14:56:20 | Mary     | <particular function> broke    |      7 |     1/5 |    7230 |
	|            		  |          | after last update, please fix! |        |         |         |
	+---------------------+----------+--------------------------------+--------+---------+---------+
	Table 1: Example user reviews dataset structure

	B) For each day, summarize the row values

	Summarize the values for each column of interest totals and averages of the various columns for each
	day of reviews, creating a new compact dataset with the following
	structure:

	+------------+-------------+----------+--------+---------+
	| Time       | ReviewCount | Rating   | Words  | Upvotes |
	+------------+-------------+----------+--------+---------+
	| 08-08-2019 |         560 |     2395 |  13230 | 230     |
	+------------+-------------+----------+--------+---------+
	Table 2: User reviews dataset after being collapsed

	"""

	# reviews + upvotes
	point_an_df = ['16.01.2019',
	'13.02.2019',
	'26.02.2019',
	'05.03.2019',
	'14.03.2019',
	'15.03.2019',
	'20.03.2019',
	'21.03.2019',
	'25.03.2019',
	'29.03.2019',
	'02.04.2019',
	'19.04.2019',
	'23.04.2019',
	'24.04.2019',
	'01.05.2019',
	'02.05.2019',
	'10.05.2019',
	'14.05.2019',
	'15.05.2019',
	'24.05.2019',
	'29.05.2019',
	'13.06.2019',
	'26.06.2019',
	'30.06.2019',
	'31.07.2019',
	'07.09.2019',
	'22.10.2019',
	'05.11.2019',
	'16.12.2019',
	'19.12.2019',
	'20.12.2019',
	'21.12.2019',
	'22.12.2019']

	only_point_an_df = ['16.01.2019',
	'13.02.2019',
	'26.02.2019',
	'05.03.2019',
	'14.03.2019',
	'15.03.2019',
	'21.03.2019',
	'25.03.2019',
	'29.03.2019',
	'19.04.2019',
	'23.04.2019',
	'24.04.2019',
	'01.05.2019',
	'02.05.2019',
	'15.05.2019',
	'30.06.2019',
	'31.07.2019',
	'07.09.2019',
	'05.11.2019',
	'16.12.2019',
	'19.12.2019',
	'20.12.2019',
	'21.12.2019',
	'22.12.2019']

	only_diff_an_df = ['16.05.2019']

	# We don't want all the columns in the samples file (.xlsx), only the ones listed below:
	# => columns that must be present in /cleaned
	SELECTED_COLUMNS_FROM_REVIEWS_DATASET = ['Rating','Review','Likes']

	def __init__(self, filename=None):
		''' When initialising we first check if we samples already exist for the given filename,
		after the sampling procedure, a .csv file is stored in the /data/reviews/ directory.

		[ 1 ] Initialisation procedure
		If samples do not exist, we carry out the sampling procedure. First we import data:
		self.cleaned_df 	- Dir: /data/cleaned	Usage: file to sample from
		self.collapsed_df 	- Dir: /data/collapsed	Usage: make overview sheet, sample days
		self.exported_df 	- Dir: /data/exported	Usage: fed to tadf

		[ 2 ] Preparation steps
		Before usage, you have to prepare:
		class variables - only_point_an_df, only_diff_an_df, poing_an_df; get from tadf
		(hint: make use of lib/tadf_exporter.py to create a file compatible with tadf)

		[ 3 ] Steps to create the samples file
			1. set 'self.converted_tadf_indices' variable, from class variables self.only_point_an_df (etc.)
			(this changes the datatime format from '%d.%m.%Y' to '%Y-%m-%d')
			
			2. initialise new list 'self.SELECTED' which will hold all the indices (dates) from the 'self.collapsed_df'
			
			3. initialise new DataFrame 'self.SAMPLED_REVIEWS' which will hold all reviews we extract from
			'self.cleaned_df', based on the datetime indices we have in self.collapsed_df

			4. self.WORK() (main)
			Here we select 10 indices (dates) from each subset: peaks, valleys, tadf, and random from the 'self.collapsed_df'
			dataset.

			5. self.collect_reviews()
			This is called once we have a list of 40 indices. Extracts the reviews on the given dates (indices),
			minimum 100 reviews per date (to a total of 4000 reviews in the resulting sample file).

			6. self.build_xlsx()


		'''

		if not filename:
			raise IOError('Must provide name of .csv file in directory /reviews to sample from!')
			exit(0)

		self.filename = filename[:-4]
		try:
			SAMPLES_FILE_ABSOLUTE = REVIEWS_DIR + filename[:-4] + "_samples.csv"
			samples_dataframe = pd.read_csv(SAMPLES_FILE_ABSOLUTE, index_col='index', parse_dates=True)
			self.SAMPLED_REVIEWS = samples_dataframe[self.SELECTED_COLUMNS_FROM_REVIEWS_DATASET].copy()
			self.SELECTED = list(self.SAMPLED_REVIEWS.index.map(lambda t: t.date().strftime('%Y-%m-%d')).unique())
			samples_exist = True
			print("✅ Samples exist, skipping sampling..")
		except Exception as e:
			samples_exist = False
			print(f'Samples do not exist, {str(e)}')

		CLEANED_FILE_ABSOLUTE = CLEANED_DIR + filename[:-4] + "_cleaned.csv"
		COLLAPSED_FILE_ABSOLUTE = COLLAPSED_DIR + filename[:-4] + "_collapsed.csv"
		EXPORTED_FILE_ABSOLUTE = EXPORTED_DIR + filename[:-4] + "_exportable.csv"
		self.collapsed_df = pd.read_csv(COLLAPSED_FILE_ABSOLUTE, index_col='index', parse_dates=True)
		self.cleaned_df = pd.read_csv(CLEANED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True)
		dateparse = lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M')
		self.exported_df = pd.read_csv(EXPORTED_FILE_ABSOLUTE, index_col='Time', parse_dates=True, date_parser=dateparse)

		verified = self.verify_equality_of_dataframes_on_date('2019-05-05', self.exported_df, self.cleaned_df)
		if not verified:
			raise IOError('Mismatch')
			exit(0)

		if not samples_exist:
			self.tadf_indices = self.only_point_an_df + self.only_point_an_df + self.point_an_df
			self.converted_tadf_indices = self.convert_datetime_indices()
			self.SELECTED = []
			self.SAMPLED_REVIEWS = pd.DataFrame(columns=self.cleaned_df.columns)
			self.WORK()
			self.SAMPLED_REVIEWS = self.SAMPLED_REVIEWS[self.SELECTED_COLUMNS_FROM_REVIEWS_DATASET].copy()
			self.SAMPLED_REVIEWS.index.name = 'index'
			self.SAMPLED_REVIEWS.to_csv(f'data/reviews/{filename[:-4]}_samples.csv')

		self.overview = self._build_overview_sheet()
		self.collect_reviews()
		self.build_xlsx()

	def verify_equality_of_dataframes_on_date(self, test_date, exported_df, cleaned_df):
		''' Keyword arguments:
		date - a date e.g., '05-05-2019'
		exported_df - user-reviews dataset that has undergone TSADFExporter (lib/tsadf_exporter.py)
		cleaned_df - a user-reviews dataset that has undegone Cleaner (lib/cleaner.py)

		Returns whether or not values in the two dataset match at a given date,
		they really should. If not, results cannot be trusted, as the exported_df
		was made based on the cleaned_df.
		'''
		if exported_df is None or cleaned_df is None:
			print('Must provide two dataframes, exported and cleaned.')
			return

		cleaned_interactions = int(len(cleaned_df.loc[test_date]) + sum(cleaned_df.loc[test_date].Likes))
		exported_interactions = int(exported_df.loc[test_date].Interactions)

		mismatch_counter = 0
		# 1. Go through each of 365 days
		for idx, row in exported_df.iterrows():
			date = str(idx).split(" ")[0]
			# 2. For each day, make sure Interactions align
			exp_interactions = row.Interactions
			cle_interactions = int(len(cleaned_df.loc[date]) + sum(cleaned_df.loc[date].Likes))
			if exp_interactions != cle_interactions:
				mismatch_counter += 1
				print(mismatch_counter, date+'\t', '(c) '+str(cle_interactions), '(e) '+str(exp_interactions))
		
		if mismatch_counter == 0:
			print("Perfect match between cleaned and exported dataframes.")

		if cleaned_interactions != exported_interactions:
			print(f'Mismatch between Cleaned ({cleaned_interactions}) and Exported ({exported_interactions}) DataFrames on date: {test_date}')
			print('Exported index: %s' % exported_df.index[0])
			print('Cleaned index: %s' % cleaned_df.index[0])
			print('=======cleaned info=======')
			print(cleaned_df.info())
			print('=======cleaned info=======')
			print('\n')
			print('=======exported info=======')
			print(exported_df.info())
			print('=======exported info=======')
			return False
		else:
			return True


	def WORK(self):
		"""Steps:
		1. Select indices from peaks, valleys, random, and tadf from collapsed dataset
		2. Collect the reviews on the selected indices from the cleaned dataset
		3. Process, manipulate; drop columns, change column names, etc.
		4. Create overview sheet
		5. Pipe the reviews into individual sheets in .xlsx book
		"""

		self.append_index_collapsed(self.converted_tadf_indices, 'TADF')
		if len(self.SELECTED) != 10:
			print("Failed to select 10 indices from tadf values")
		self.append_peaks()
		if len(self.SELECTED) != 10*2:
			print("Failed to select 10 indices from peak values")
		self.append_valleys()
		if len(self.SELECTED) != 10*3:
			print("Failed to select 10 indices from valley values")
		self.append_random()
		if len(self.SELECTED) != 10*4:
			print("Failed to select 10 indices from random values")

		for idx in self.SELECTED:
			self._verify_date_selection(idx)

		if len(self.SELECTED) != 40:
			print(f'\033[93mWrong amount of indices selected {len(self.SELECTED)} != 40\033[0m')
		else:
			print(f'\033[92mSelected 40 indices\033[0m')

	def append_peaks(self, column='Interactions'):
		''' Takes all values in provided column and creates an nparray.
		Once it is an nparray, we use argsort to locate the highest values.
		Then, for each high value we find the index of the row containing
		that value in the dataframe.
		Finally, we call 'append_index_collapsed' with the list of indices.
		'''
		column_counts = self.collapsed_df[column]
		column_nparray = np.array(column_counts)
		largest_values = column_nparray[np.argsort(column_nparray)[-50:]]
		peaks = sorted(largest_values, reverse=True) # sort descending

		# find the indices via the largest values
		highest_indices = []
		for peak_value in peaks:
			desired_index = self.collapsed_df.index[self.collapsed_df[column] == peak_value][0]
			desired_index_string = desired_index.strftime('%Y-%m-%d')
			highest_indices.append(desired_index_string)
		self.append_index_collapsed(highest_indices, 'PEAKS')

	def append_valleys(self, column='Interactions'):
		''' Takes all values in provided column and creates an nparray.
		Once it is an nparray, we sort the values in descending order,
		and select the last (lowest) 10 values.

		We then look through the dataframe for the rows that contain
		these values, and extract their indices. Finally we call
		'append_index_collapsed' with the list of indices.
		'''
		column_counts = self.collapsed_df[column]
		column_nparray = np.array(column_counts)
		cnpa = sorted(column_nparray, reverse=False)
		lowest_values = cnpa[:10]

		lowest_indices = []
		for low_value in lowest_values:
			desired_index = self.collapsed_df.index[self.collapsed_df[column] == low_value][0]
			desired_index_string = desired_index.strftime('%Y-%m-%d')
			lowest_indices.append(desired_index_string)
		self.append_index_collapsed(lowest_indices, 'VALLEYS')

	def append_random(self):
		import random
		random_date_objects = random.sample(list(self.collapsed_df.index), 100)
		Sampler._verify_dateformat('%Y-%m-%d', random_date_objects[0])
		random_date_strings = [r.strftime('%Y-%m-%d') for r in random_date_objects]
		self.append_index_collapsed(random_date_strings, 'RANDOM')
		
	def get_strat_sample(self, date, column='Rating'):
		from sklearn.model_selection import StratifiedShuffleSplit

		reviews = self.cleaned_df.loc[date]
		self._verify_date_selection(date)
		values = reviews[column]
		
		one_pct = len(reviews) / 100
		pct_100 = round((100 / one_pct)/100, 2)
				
		try:
			sss = StratifiedShuffleSplit(n_splits=1, test_size=pct_100, random_state=0)
			for train_index, test_index in sss.split(reviews, values):
				X_train, X_test = reviews.iloc[train_index], reviews.iloc[test_index]
				y_train, y_test = values.iloc[train_index], values.iloc[test_index]
			return X_test
		except ValueError:
			print(f"\033[93mInsufficient class labels in one of your classes to keep the data split ratio equal to test_size, 100 random reviews will be returned.\033[0m")
		return reviews.sample(100)

	def collect_reviews(self):
		''' This method operates on the 'self.SELECTED' field, which is a list
		containing 40 unique dates from the 'self.collapsed_df'.

		The goal of the method is to -- for each date from the collapsed dataframe --
		go into the cleaned dataframe and extract at least 100 reviews from the user
		reviews dataset (self.cleaned_df).

		For each date, it performs a verification step to make sure there's a match
		between the selection in the collapsed and exported dataframe. It does this
		because the date-formats of the indices in the datasets are initially not the
		same and can be sensitive to the dates provided e.g., '01-05-2020', or '05-01-2020'.

		After the verification, it collects all reviews on the given date, and checks
		if the total amount of reviews is more than 100. Depending on the result:
			More than 100 reviews for date: perform stratified sample based on rating
			Less than or equal to 100 reviews: take all reviews
		'''
		lt_100_counter = 0
		for date in self.SELECTED:
			# date = date.strftime('%Y-%m-%d').replace('00','').replace(':','') if type(date) != str else date
			self._verify_date_selection(date)
			data = self.cleaned_df.loc[date]
			if len(data) <= 100:
				print(f'For date: {date} found LT 100 reviews ({len(data)}). Selecting all')
				lt_100_counter += 1
				sample = data
			else:
				__one_pct = len(data) / 100
				__pct_100 = round((100 / __one_pct)/100, 4)
				print(f'[Amount selected: {len(self.SAMPLED_REVIEWS)}] (Stratify on rating) Found {len(data)} reviews on {date}, collecting {__pct_100 * len(data)} reviews')
				sample = self.get_strat_sample(date)
				
			# iterate over each row in the selected sample
			for idx, row in sample.iterrows():
				self.SAMPLED_REVIEWS.loc[idx] = row

		if lt_100_counter != 0:
			print(f"{lt_100_counter}/40 days did not have 100 reviews")
		else:
			print("All days had at least 100 reviews")
		print(f"{len(self.SAMPLED_REVIEWS)} reviews selected from {len(self.SELECTED)} unique dates")

	def append_index_collapsed(self, indices, title=None):
		""" This method takes an arbitrary amount of indices
		and appends the first 10 unique values to an array.

		It is a good idea to provide more than 10 indices,
		as an index (date) may have been selected before.
		"""
		if len(self.SELECTED) >= 40:
			print("Not adding more indices! Already at 40!")
			return

		print('\n==== Appending =====') if title == None else print(f'\n==== Appending {title} ====')
		
		duplicates = []
		counter = 0

		# TODO: Randomize the selection of indices,
		# not just start at the top with a for-loop
		# like we're doing here.
		for idx in indices:
			if counter >= 10:
				break
			elif idx in self.SELECTED:
				duplicates.append(idx)
				continue
			else:
				if type(idx) != str:
					idx = str(idx)
				elif type(idx) == str:
					idx

				print(f'{title}: appending {idx} ({type(idx)}')
				self.SELECTED.append(idx)
				counter += 1
		
		if counter < 10:
			print("\033[91mWas not able to append 10 values\033[0m")
		else:
			print(f"{self.SELECTED}")
			print(f'==== Completed (+{counter}) =====')


	def convert_datetime_indices(self):
		converted_tadf_indices = []
		for date_string in self.tadf_indices:
			date_object = datetime.strptime(date_string, '%d.%m.%Y')
			new_date_string = date_object.strftime('%Y-%m-%d')
			new_date_object = datetime.strptime(new_date_string, '%Y-%m-%d')

			if not date_object.year == new_date_object.year:
				print('something went to shit mixing with the dates!')
			if not date_object.month == new_date_object.month:
				print('something went to shit mixing with the dates!')
			if not date_object.day == new_date_object.day:
				print('something went to shit mixing with the dates!')
			
			converted_tadf_indices.append(new_date_string)
		print("Converted dates for tadf dataset.. No longer tadf compliant.")
		return converted_tadf_indices

	@staticmethod
	def _verify_dateformat(expected_format, date_obj):
			raw_format = '%Y-%m-%dT%H:%M:%sTz'
			clean_ex, clean_form = '2019-12-31 23:51:59', '%Y-%m-%d %H:%M:%S'
			tadf_ex, tadf_form = '26.03.2020 00:00', '%d.%m.%Y %H:%M'

			if expected_format == '%Y-%m-%d':
				y, m, d = date_obj._date_repr.split("-")[0],date_obj._date_repr.split("-")[1], \
							date_obj._date_repr.split("-")[2]
				if date_obj.year != int(y) or date_obj.month != int(m) or date_obj.day != int(d):
					print(f"\033[91mDate format not verified, expected format: {expected_format}, dateobj: {date_obj}\033[0m")
					return False
			return True

	def _verify_date_selection(self, date):
		"""Verfies that, for each selected index which we take from the collapsed
		dataset, the values present in the collapsed dataset on a given date
		match the values found in the cleaned dataset - from which we built the
		collapsed dataset.
		"""

		collapsed_reviewcount = self.collapsed_df.loc[date]['ReviewCount']
		collapsed_interactions = int(self.collapsed_df.loc[date]['Interactions'])
		cleaned_reviewcount = len(self.cleaned_df.loc[date])
		exported_interactions = int(self.exported_df.loc[date]['Interactions'])

		if collapsed_reviewcount != cleaned_reviewcount:
			print(f"\033[91mCollapsed/Cleaned comparison mismatch: {collapsed_reviewcount}/{cleaned_reviewcount} {str(date)}\033[0m")
			exit(0)
		elif collapsed_interactions != exported_interactions:
			print(f"\033[91mCollapsed/Exported comparison mismatch: {collapsed_interactions}/{exported_interactions} {str(date)}\033[0m")
			exit(0)

	def _verify_strat_sample(self, date):
		# Investigate the distribution of values in the sample
		strat_review_sample = self.get_strat_sample(date)
		s_distribution = strat_review_sample.value_counts() / len(strat_review_sample)

		# Investigate the distribution of values in the day of
		# reviews from which the sample was taken, they should be
		# very similar
		day_of_reviews = self.cleaned_df.loc[date]
		d_distribution = day_of_reviews.value_counts() / len(day_of_reviews)

		if len(s_distribution) != len(d_distribution):
			print('\033[91mStrat sample verification failed, mismatched columns?\033[0m')
			print(s_distribution+'\n')
			print(d_distribution)
			return False

		for i in range(1,1+len(s_distribution)):
			strat, day = s_distribution.loc[i], d_distribution.loc[i]
			diff = strat - day
			print(diff)
			if abs(diff) > 1:
				print(f'\033[93mDifference larger than 1, not the best..\033[0m')
				return False
		return True

	def _build_overview_sheet(self):
		print('Putting together the overview sheet of the .xlsx')
		sources = 10*['TSADF'] + 10*['PEAKS'] + 10*['VALLEYS'] + 10*['RANDOM']

		REVIEWS_COUNT = []
		REVIEWS_NEGATIVE_COUNT = []
		REVIEWS_POSITIVE_COUNT = []
		REVIEWS_NEGATIVE_PERCENTAGE = []

		UPVOTES_COUNT = []
		UPVOTES_POSITIVE_COUNT = []
		UPVOTES_NEGATIVE_COUNT = []
		UPVOTES_NEGATIVE_PERCENTAGE = []
		UPVOTES_NEGATIVE_AVERAGE = []
		# UPVOTES_NEGATIVE_MEDIAN = []

		UPVOTES_POSITIVE_AVERAGE = []
		# UPVOTES_POSITIVE_MEDIAN = []

		INTERACTIONS_COUNT = []
		INTERACTIONS_NEGATIVE_COUNT = []
		INTERACTIONS_POSITIVE_COUNT = []
		INTERACTIONS_NEGATIVE_PERCENTAGE = []

		co = 0
		print(f'Will iterate over {len(self.SELECTED)} unique dates.')
		for d in self.SELECTED:
			reviews = self.cleaned_df.loc[d]

			total = len(reviews)
			REVIEWS_COUNT.append(total)

			negative_reviews = reviews[reviews.Rating <= 3]
			negative_review_count = len(negative_reviews)
			REVIEWS_NEGATIVE_COUNT.append(negative_review_count)
			
			positive_reviews = reviews[reviews.Rating > 3]
			positive_review_count = len(positive_reviews)
			REVIEWS_POSITIVE_COUNT.append(positive_review_count)

			neg_r = negative_review_count / (total/100)
			REVIEWS_NEGATIVE_PERCENTAGE.append(round((neg_r/100), 2))
			
			all_upvotes = sum(reviews.Likes)
			UPVOTES_COUNT.append(all_upvotes)

			negative_up = sum(reviews[reviews.Rating <= 3].Likes)
			UPVOTES_NEGATIVE_COUNT.append(negative_up)

			positive_up = sum(reviews[reviews.Rating > 3].Likes)
			UPVOTES_POSITIVE_COUNT.append(positive_up)
			
			upv_neg_pcnt = negative_up / (all_upvotes/100)
			UPVOTES_NEGATIVE_PERCENTAGE.append(round((upv_neg_pcnt/100),2))
			
			avg_num = s.mean(negative_reviews.Likes)
			UPVOTES_NEGATIVE_AVERAGE.append(round(avg_num, 1))
			
			# med_num = s.median(negative_reviews.Likes)
			# UPVOTES_NEGATIVE_MEDIAN.append(med_num)

			avg_pos_num = s.mean(positive_reviews.Likes)
			UPVOTES_POSITIVE_AVERAGE.append(round(avg_pos_num, 1))

			# med_pos_num = s.median(positive_reviews.Likes)
			# UPVOTES_POSITIVE_MEDIAN.append(round(med_pos_num, 1))

			# all_interactions = total * all_upvotes
			all_interactions = total + all_upvotes
			INTERACTIONS_COUNT.append(all_interactions)
			
			# neg_interactions = negative_review_count * negative_up
			neg_interactions = negative_review_count + negative_up
			INTERACTIONS_NEGATIVE_COUNT.append(neg_interactions)
			
			# pos_interactions = positive_review_count * positive_up
			pos_interactions = positive_review_count + positive_up
			INTERACTIONS_POSITIVE_COUNT.append(pos_interactions)
			
			percent_negative_interactions = neg_interactions / (all_interactions / 100)
			INTERACTIONS_NEGATIVE_PERCENTAGE.append(round(percent_negative_interactions/100, 2))
			
			co += 1
			
		my_dict = {
			'Source' : sources,
			'Date' : self.SELECTED,
			'Interactions' : 	INTERACTIONS_COUNT,
			'I% <= 3' : 		INTERACTIONS_NEGATIVE_PERCENTAGE,
			'I <= 3' : 			INTERACTIONS_NEGATIVE_COUNT,
			'I > 3' : 			INTERACTIONS_POSITIVE_COUNT,

			'Reviews' : 		REVIEWS_COUNT,
			'Rev% <= 3' : 		REVIEWS_NEGATIVE_PERCENTAGE,
			'R <= 3' : 			REVIEWS_NEGATIVE_COUNT,
			'R > 3' : 			REVIEWS_POSITIVE_COUNT,

			'Upvotes' : 		UPVOTES_COUNT,
			'Up <= 3' : 		UPVOTES_NEGATIVE_COUNT,
			'Up > 3' : 			UPVOTES_POSITIVE_COUNT,
			'Up% <= 3' : 		UPVOTES_NEGATIVE_PERCENTAGE,
			'Avg <= 3' : 		UPVOTES_NEGATIVE_AVERAGE,
			'Avg > 3' : 		UPVOTES_POSITIVE_AVERAGE
			# 'Med <= 3' : 		UPVOTES_NEGATIVE_MEDIAN,
			# 'Med > 3' : 		UPVOTES_POSITIVE_MEDIAN
		}
		overview_df = pd.DataFrame(my_dict)
		self._verify_overview(overview_df)
		print("✅ Overview built")
		return overview_df

	def _verify_overview(self, overview_df):
		''' For each of the dates in our overview, the values
		should be the same in both overview and exportable
		(and collapsed)
		'''
		for i, row in overview_df.iterrows():
			overview_interactions = row.Interactions
			exported_interactions = self.exported_df.loc[row.Date].Interactions
			collapsed_interactions = self.collapsed_df.loc[row.Date].Interactions
			if overview_interactions != exported_interactions:
				print(f'Overview/Exported mismatch {overview_interactions}/{exported_interactions}')
				return False
			if overview_interactions != collapsed_interactions:
				print(f'Overview/Collapsed mismatch {overview_interactions}/{collapsed_interactions}')
				return False
		print("✅ All dates were verified")

	def build_xlsx(self):
		""" Operates on the 'self.SELECTED_REVIEWS' dataframe, which contains 4000 reviews,
		and only columns ['Rating', 'Review', 'Likes'].

		Creates an overview sheet, and one more sheet per day.

		At the top of each sheet, we put the 10 most upvoted reviews on the given date,
		taken from the cleaned dataframe (all reviews considered, not only samples).
		"""
		print(f'Putting together .xlsx')
		import xlsxwriter
		import string
		# Make a copy of the DataFrame containing all the reviews sampled from the 40 days
		sel = self.SAMPLED_REVIEWS.copy()
		sel.index = sel.index.tz_localize(None) # Remove timezone information
		
		# Make new ExcelWriter
		writer = pd.ExcelWriter(f'data/xlsx/{self.filename}_bible.xlsx', engine='xlsxwriter')
		workbook = writer.book

		# Write the overview DataFrame to sheet 'Overview'
		self.overview.to_excel(writer, sheet_name='Overview')
		# Grab the 'Overview' sheet and adjust some settings, column width
		worksheet = writer.sheets['Overview']
		worksheet.set_column('C:C', 10) # Date
		worksheet.set_column('D:D', 10) # Inter
		worksheet.set_column('H:H', 10) # Reviews
		worksheet.set_column('L:L', 10) # Up
		# add cell references at the end of each Overview row
		alert_format = workbook.add_format({
			'bold' : True,
			'underline' : True,
			'color': '#8b0000',
			'align' : 'center',
			'align' : 'vcenter',
			'font_size' : 14
		})

		worksheet.write('R1', 'Alert?', alert_format)
		worksheet.set_column('R:R', 13)

		counter = 1
		sheet_names = []

		for date in self.SELECTED:
			sheet_name = 'Day_' + str(counter)
			sheet_names.append(sheet_name)

			# sheet_of_reviews = pd.concat([ten_most_upvoted_reviews, sel.loc[date]])
			sheet = sel.loc[date]
			# 1. Sort by Rating
			sheet = sheet.sort_values('Rating', ascending=True)

			# 2. Sort by Upvotes
			rat_1 = sheet[sheet.Rating == 1].sort_values('Likes', ascending=False)
			rat_2 = sheet[sheet.Rating == 2].sort_values('Likes', ascending=False)
			rat_3 = sheet[sheet.Rating == 3].sort_values('Likes', ascending=False)
			rat_4 = sheet[sheet.Rating == 4].sort_values('Likes', ascending=False)
			rat_5 = sheet[sheet.Rating == 5].sort_values('Likes', ascending=False)
			
			# 3. Add top 10 most upvoted to the top
			top_upvoted = self.get_largest_values(self.cleaned_df, date, 10, 'Likes')
			top_upvoted_idx = self.get_indices_via_upvotes(self.cleaned_df, date, top_upvoted, 'Likes')
			ten_most_upvoted_reviews = self.cleaned_df.loc[top_upvoted_idx]
			ten_most_upvoted_reviews.index = ten_most_upvoted_reviews.index.tz_localize(None)
			# Column count has to align with that in the samples
			ten_most_upvoted_reviews = ten_most_upvoted_reviews[self.SELECTED_COLUMNS_FROM_REVIEWS_DATASET]
			sheet = pd.concat([ten_most_upvoted_reviews, rat_1, rat_2, rat_3, rat_4, rat_5])

			before_dupe_count = len(sheet)
			sheet = sheet.drop_duplicates()
			sheet.to_excel(writer, sheet_name=(sheet_name))

			# Write the corresponding row from the Overview sheet
			overview_row = self.overview[self.overview.Date == date]
			bold_format = workbook.add_format()
			bold_format.set_bold()
			LETTERS = list(string.ascii_uppercase)[5:] # Start at 'F'
			source_date = [0,1]
			interactions = [2,3,4,5]
			reviews = [6,7,8,9]
			upvotes = [10, 11, 12, 13, 14, 15]
			for i, column in enumerate(overview_row.columns):
				worksheet = writer.sheets[sheet_name]

				if i in source_date:
					colname = f'{LETTERS[source_date.index(i)]}'
					colname_pos, value_pos = 4,5
				elif i in interactions:
					colname = f'{LETTERS[interactions.index(i)]}'
					colname_pos, value_pos = 7, 8
				elif i in reviews:
					colname = f'{LETTERS[reviews.index(i)]}'
					colname_pos, value_pos = 10, 11
				elif i in upvotes:
					colname = f'{LETTERS[upvotes.index(i)]}'
					colname_pos, value_pos = 13, 14

				worksheet.write(f'{colname}{colname_pos}', str(column), bold_format)
				worksheet.write(f'{colname}{value_pos}', overview_row[column].values[0])
			# Lastly, make a cell into we write 'Alert'..
			worksheet.write('H2', 'Alert?', alert_format)
			worksheet.write('H3', 'Not assigned')
			worksheet.set_column('H:H', 20)

			# Pick up the worksheet via its name and make adjustments to style
			worksheet = writer.sheets[sheet_name]
			# Set width of review column
			cell_format = workbook.add_format()
			cell_format.set_text_wrap()
			worksheet.set_column('C:C', 75, cell_format)

			# 'Back to overview' link, in every sheet
			link_format = workbook.add_format({
				'bold' : True,
				'italic' : True,
				'color' : 'blue',
				'underline' : True,
				'align' : 'center',
				'align' : 'vcenter',
				'font_size' : 14
			})

			worksheet.write_url('F3', 'internal:Overview!A1')
			worksheet.write('F3', 'Back to Overview', link_format)
			worksheet.set_column('F:F', 20) # Overview link
			worksheet.set_column('G:G', 10) # Date
			worksheet.set_column('H:H', 10) # Interactions

			counter += 1

		# Link to individual sheets
		worksheet = writer.sheets['Overview']
		for i, sheet_name in enumerate(sheet_names):
			row = i + 1
			# Create a link
			worksheet.write_url(f'A{row+1}', f'internal:Day_{row}!E1')
			# Add text to the cell
			worksheet.write(f'A{row+1}', f'Day {row}', link_format)

		# Open up the Overview sheet, make reference to 'Alert' 
		days = list(range(2,42)) # Starts at 2, ends at 41
		dates = self.SELECTED
		ddmap = list(zip(days, dates))

		# In the overview sheet, for each R<Day>, have a reference to H6 in each daily sheet
		for _day in days:
			worksheet.write(f'R{_day}', f'=Day_{_day-1}!H3') # Off by 1

		writer.save()
		print("✅ .xlsx built")

	def get_largest_values(self, dataframe, date, amount, column):
		all_values = np.array(dataframe.loc[date][column])
		largest = all_values[np.argsort(all_values)[-amount:]]
		return sorted(largest, reverse=True)

	def get_indices_via_upvotes(self, dataframe, date, upvote_list, column):
		"""Returns the indices of all reviews with upvote count
		matching the values found in the provided upvote_list.

		As there are several reviews that may share the same upvote
		count, we simply return the first.
		"""
		indices_found, to_find_len, found_len = [], len(upvote_list), 0

		for upvote in upvote_list:
			for idx, row in dataframe.loc[date].iterrows():
				if row[column] == upvote:
					indices_found.append(idx)
					found_len += 1
					break

		# if found_len < to_find_len:
		# 	# print(f'\033[91mOnly {found_len} out of {to_find_len} indices were found via upvotes!\033[0m')
		# elif found_len == to_find_len:
		# 	# print(f'\033[92mFound {found_len}/{to_find_len} indices via upvotes.\033[0m')
		# else:
		# 	print(f'\033[1mFound: {found_len}, Asked to find: {to_find_len}\033[0m')
		return indices_found
