import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import time
import pdb

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

def delete_stop_words(corpus):
	stop_words = set(stopwords.words('english'))
	filtered_list = [w for w in corpus if not w.lower() in stop_words]

	return filtered_list

def lemmatize(corpus):
	lemmatizer = WordNetLemmatizer()
	lemmatized_list = [lemmatizer.lemmatize(w) for w in corpus]

	return lemmatized_list

def stem(corpus):
	stemmer = PorterStemmer()
	stemmed_list = [stemmer.stem(w) for w in corpus]

	return stemmed_list

def one_gram_frequency(corpus, top_n):
	data_analysis = nltk.FreqDist(corpus)
	filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 1])

	sort_orders = sorted(filter_words, key=filter_words.get, reverse=True)

	for i, key in enumerate(sort_orders):
		print("%s: %s" % (key, filter_words[key]))
		if i>top_n:
			return

def make_corpus(title_list):

	title_list_clean = []
	for title in title_list:
		if str(title) == 'nan':
			continue
		else:
			title_list_clean.append(title)
	corpus = " ".join(title_list_clean)
	return corpus

def get_tags(text):
	text_list = []
	text_list.append(text)
	tokenizer_nltk = RegexpTokenizer(r'\w+')
	corpus = make_corpus(text_list)
	tokens = stem(lemmatize(delete_stop_words(tokenizer_nltk.tokenize(corpus))))
	return tokens

def include_title(words_to_include, title):
	if pd.isna(title):
		return 0
	for word in words_to_include:
		if word in title:
			return 1
	return 0

def add_tags(data):
    tags_list = []
    words_to_include = ["bra", "lingerie", "shorts", "legging", "hijab", "coat", "brush", "vest", "shirt", "pant", "trouser", "jean", "jacket", "pyjama", "sweatshirt", "denim", "shapewear", "waistcoat", "cargo", "blazer", "capri", "top", "dhoti", "kurta", "dress", "saree", "choli", "lehenga", "dupatta", "palazzo", "nighti", "salwar suit", "salwar", "anarkali", "midi", "maxi", "blous", "churidar", "gown", "skirt", "jumpsuit", "shrug", "camisol", "patiala", "sharara", "bikini"]
    for no, row in data.iterrows():
        if no%10000 == 0:
            print('Done with',no,'items')
    # if include_title(words_to_include, row.title):
        if not pd.isna(row.title):
            tags_list.append(get_tags(row.title.replace(str(row.brand), "")))
        else:
            tags_list.append(get_tags(row.title))
    # else:
        # data.drop(data.index[no], inplace=False)

    data["tags"] = tags_list
