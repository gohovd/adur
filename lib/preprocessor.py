import logging
import time

log = logging.getLogger(__name__)

class Preprocessor:
	'''

	This class produces an additional two datasets:
		1. Preprocessed (ready for classification) dataset, from the cleaned dataset
		2. Labeled data, for training the classifier on category and sentiment

	Some columns may be more valuable with respect to classifier accuracy than others,
	it should be easy to plug-and-play another column, for experiment.

	'''

	def __init__(self):
		log.info('Initialized preprocessor.')
