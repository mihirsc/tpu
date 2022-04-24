from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

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

def clean_attributes(attributes):
    tokenizer_nltk = RegexpTokenizer(r'\w+(?:-\w+)*')
    attributes = [a for a in attributes if a not in ['no non-textile material', 'no special manufacturing technique', 'no waistline']]
    attributes_joined = ' '.join(attributes)
    tokens = stem(lemmatize(delete_stop_words(tokenizer_nltk.tokenize(attributes_joined))))
    return tokens