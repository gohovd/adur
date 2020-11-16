import logging
import os
import pandas as pd

PATH_LIST = os.path.abspath(__file__).split(os.sep)
ROOT_DIR = PATH_LIST[0:len(PATH_LIST)-2]

CLEANED_DIR = "/".join(ROOT_DIR) + "/data/cleaned/"
COLLAPSED_DIR = "/".join(ROOT_DIR) + "/data/collapsed/"
EXPORTABLE_DIR = "/".join(ROOT_DIR) + "/data/exportable/"

import time
from datetime import datetime
from calendar import monthrange
import statistics as s
import pandas as pd

log = logging.getLogger(__name__)

class TSADFExporter:
	'''Performs necessary step to make the dataset TSADF-compliant.

	DatetimeIndex formats:
	raw - 2020-03-26T10:44:31.700Z (%Y-%m-%dT%H:%M:%sTz)
	clean - 2019-12-31 23:51:59 (%Y-%m-%d %H:%M:%S)
	tsadf - 26-03-2020 00:00 (%d.%m.%Y %H:%M)

	Produces two datasets:
	data/collapsed/<filename>_collapsed.csv
	data/exportable/<filename>_exportable.csv (copy of collapsed, index changed, only one column)
	'''

	def __init__(self, reviews):
		# First load the 'cleaned' dataset, from this we make new datasets:
		# - collapsed (/collapsed directory)
		# - exportable (/exportable directory)
		CLEANED_FILE_ABSOLUTE = CLEANED_DIR + reviews[:-4] + "_cleaned.csv"
		log.info("Absolute path of the cleaned dataset: %s" % CLEANED_FILE_ABSOLUTE)
		self.dataframe = pd.read_csv(CLEANED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True)

		# Make dataset of 365 days, each row is a day with sum-totals and averages
		# of the reviews written on that day:
		# - total number of reviews, upvotes, words used, interactions, ...
		self.collapsed = TSADFExporter.build(self.dataframe, 2019)
		collapsed_filename = reviews[:-4] + "_collapsed.csv"
		self.collapsed.to_csv(COLLAPSED_DIR + collapsed_filename, columns = self.collapsed.columns)

		# Make a TADF compliant dataset of the user reviews provided dataset
		# this changes datetime-index of all rows to %d.%m.%YYYY
		self.tadf_compliant = TSADFExporter.build(self.dataframe, 2019, True)
		exportable_filename = reviews[:-4] + "_exportable.csv"
		# self.tadf_compliant.to_csv(EXPORTABLE_DIR + exportable_filename, columns = ['Time', 'Interactions'])
		# Make a TADF compliant "minimal" dataset of the user reviews provided
		# this retains only the "interactions" column, also changes the datetime-format
		self.tadf_compliant_minimal = self.tadf_compliant[['Interactions']].copy()
		self.tadf_compliant_minimal.to_csv(EXPORTABLE_DIR + exportable_filename, columns = self.tadf_compliant_minimal.columns)

	@staticmethod
	def build(dataframe, year, tadf_compliant=False):
		''' This method takes a user-reviews dataset, and a year,
		and returns a new dataset (DataFrame) of 365 days/rows,
		where each day/row shows the sum-totals and averages
		of the provided user-reviews dataset:

		Day | Words | Likes | Reviews (see all below)
		5	  1000	  754	  400
		
		The values show for each day the total number of likes,
		reviews, words, and more.

		'''
		print('Building collapsed dataset..') if not tadf_compliant else print('Building TADF compliant dataset..')
		start = time.time()
		collapsed = {
			'Time' if tadf_compliant else 'index' : [],
			'ReviewCount': [],
			'Interactions' : [],

			'Words' : [],
			'Chars' : [],
			'Likes' : [],
			'Rating' : [],
			'Avg_Words' : [],
			'Avg_Chars' : [],
			'Avg_Likes' : [],
			'Avg_Rating' : [],

			'Neg_Interactions' : [],
			'Pos_Interactions' : []
		}
		for month in range(1, 13):
			for day_of_month in range(1, monthrange(year, month)[1]+1): # starts at 0, so we add 1
				index_string = f'{year}-{month}-{day_of_month}'
				datetimeindex = datetime.strptime(index_string + ' 00:00', '%Y-%m-%d %H:%M')

				if tadf_compliant:
					# Then we need trailing 00:00 in index
					datetimeindex = datetimeindex.strftime('%d.%m.%Y') + ' 00:00'
					collapsed['Time'].append(datetimeindex)
				else:
					collapsed['index'].append(datetimeindex)

				df_slice = dataframe.loc[index_string]
				reviewcount = len(df_slice)
				interactions = int(reviewcount) + sum(df_slice.Likes)

				if not str(interactions).isnumeric() or type(interactions) is not int:
					print(interactions)

				collapsed['ReviewCount'].append(reviewcount)
				collapsed['Interactions'].append(interactions)

				collapsed['Words'].append(sum(df_slice['Words']))
				collapsed['Chars'].append(sum(df_slice['Chars']))
				collapsed['Likes'].append(sum(df_slice['Likes']))
				collapsed['Rating'].append(sum(df_slice['Rating']))

				collapsed['Avg_Words'].append(round(s.mean(df_slice['Words']), 2))
				collapsed['Avg_Chars'].append(round(s.mean(df_slice['Chars']), 2))
				collapsed['Avg_Likes'].append(round(s.mean(df_slice['Likes']), 2))
				collapsed['Avg_Rating'].append(round(s.mean(df_slice['Rating']), 2))

				neg_reviews = df_slice[df_slice.Rating <= 3]
				pos_reviews = df_slice[df_slice.Rating > 3]

				# collapsed['Neg_Interactions'].append(int(float(len(neg_reviews) * sum(neg_reviews.Likes))))
				# collapsed['Pos_Interactions'].append(int(float(len(pos_reviews) * sum(pos_reviews.Likes))))

				collapsed['Neg_Interactions'].append(len(neg_reviews) + int(sum(neg_reviews.Likes)))
				collapsed['Pos_Interactions'].append(len(pos_reviews) + int(sum(pos_reviews.Likes)))

				# print(f"Index: {datetimeindex} Count: {reviewcount} Interactions: {interactions}")

		collapsed_df = pd.DataFrame.from_dict(collapsed)
		collapsed_df = collapsed_df.set_index('index') if not tadf_compliant else collapsed_df.set_index('Time')
		print(f'completed in {round((time.time() - start),2)}s')
		return collapsed_df