import tkinter as tk
import sys
import logging
import time
import os
import configparser
from argparse import ArgumentParser

from lib.cleaner import Cleaner
from lib.preprocessor import Preprocessor
from lib.classification import Classifier

PATH_LIST = os.path.abspath(__file__).split(os.sep)
ROOT_DIR = PATH_LIST[0:len(PATH_LIST)-1]
REVIEWS_DIR = "/".join(ROOT_DIR) + "/data/reviews/"
NORMALIZED_DIR = "/".join(ROOT_DIR) + "/data/normalized/"
LABELS_DIR = "/".join(ROOT_DIR) + "/data/training/"

config = configparser.ConfigParser()
config.read('config.ini')

path = './'
exclude = []
DIRS = [x[0] for x in os.walk(path)]
for d in DIRS:
	split_d = d.split('/')
	common = list(set(split_d).intersection(exclude))
	if len(common) < 1:
		sys.path.append(d+'/')

def input():
	ap = ArgumentParser()
	ap.add_argument("-f", "--file", help="Name of file in /reviews directory", required=True)
	ap.add_argument("-y", "--year", help="Which year of the data to analyze", required=True)
	ap.add_argument("-v", "--verbose", help="Logging output verbosity", action="store_true")
	args = vars(ap.parse_args())
	return args

if __name__ == "__main__":
	args = input()

	log = logging.getLogger()

	if args['verbose']:
		log.setLevel(logging.DEBUG)
	else:
		log.setLevel(logging.INFO)

	# while log.handlers:
	# 	log.handlers.pop()
	log_handler = logging.StreamHandler()
	log_format = logging.Formatter('%(name)s %(asctime)s %(levelname)s --- %(message)s')
	log_handler.setFormatter(log_format)
	log.addHandler(log_handler)

	try:
		REVIEW_FILE = args['file']
		print(REVIEWS_DIR + REVIEW_FILE)
		if not os.path.isfile(REVIEWS_DIR + REVIEW_FILE):
			log.error('File not found')
			exit(0)

	except Exception as e:
		print('Error: {}'.format(e))
		exit(0)

	log.info(f'Started, selected reviews file is: ({REVIEW_FILE})')


'''
Pipeline steps:

No.		Class			Input								Output
1		Cleaner			/data/reviews/<file>.csv			/data/cleaned/<file>_cleaned.csv
2		Preprocessor	/data/reviews/<file>_cleaned.csv	-
3		TSADFExporter	/data/reviews/<file>_cleaned.csv	/data/exportable/<file>_exported.csv
4		Normalizer		/data/reviews/<file>_cleaned.csv	/data/normalized/<file>_normalized.csv
5		Classifier		/data/reviews/<file>_normalized.csv	-
'''

cleaner = Cleaner(
	reviews=REVIEW_FILE,
	year=args['year'],
	min_words=config.get('CLEANER', 'MIN_WORDS')
	)

preprocessor = Preprocessor(
	reviews=REVIEW_FILE,
	column='Review',
	year=args['year'],
	stem=config.get('CLASSIFIER', 'STEM'),
	lemmatize=config.get('CLASSIFIER', 'LEMMATIZE')
	)

# TODO: Implement
# 3. Classify
# NORMALIZED_REVIEWS_ABSOLUTE = NORMALIZED_DIR + REVIEW_FILE[:-4] + "_normalized.csv"
# CATEGORY_SENTIMENT_LABELS_ABSOLUTE = LABELS_DIR + "sen_cat.csv"
# clf = Classifier(
# 	reviews=NORMALIZED_REVIEWS_ABSOLUTE,
# 	labels=CATEGORY_SENTIMENT_LABELS_ABSOLUTE
# )