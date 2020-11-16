class Normalizer:
	''' Prepares dataset for text classification '''
	def __init__(self, dataframe=None, year=None, reviews=None, stem=True, lemmatize=False, column=None):

		self.reviews_file = reviews

		try:
			NORMALIZED_FILE_ABSOLUTE = NORMALIZED_DIR + reviews[:-4] + "_normalized.csv"
			self.dataframe = pd.read_csv(NORMALIZED_FILE_ABSOLUTE, index_col='Timestamp', parse_dates=True).loc[year]
			log.info("✅ Found normalized version, skipping normalization..")
			normalized_version_exists = True
			return
		except Exception as e:
			log.info(f'initialized normalizer with settings - stem {stem} lemmatize {lemmatize} column {column} year {year}')
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