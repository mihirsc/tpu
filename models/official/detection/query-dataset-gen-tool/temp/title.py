import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import pdb

def delete_stop_words(corpus, stop_words):
    return [w for w in corpus if not w in stop_words]

def lemmatize(corpus, lemmatizer):
    return [lemmatizer.lemmatize(w) for w in corpus]

def stem(corpus, stemmer):
    return [stemmer.stem(w) for w in corpus]

def make_corpus(title_list):
    title_list_clean = []
    for tag in title_list:
        if str(tag) not in ['nan' ,'&', '.']:
            title_list_clean.append(tag)

    title_list_clean = " ".join(title_list_clean).replace('men', 'man')
    return title_list_clean

def find_gender(tokens_stemmed): # returns gender if explicitly mentioned in title, empty otherwise

    women = set(['girl', 'woman', 'female'])
    men = set(['boy', 'man', 'male'])
    kid = set(['kid', 'child'])

    gender, gender_words = '', set()

    if len(women&tokens_stemmed) > 0 and len(men&tokens_stemmed) > 0:
        gender, gender_words = 'unisex', women&tokens_stemmed | men&tokens_stemmed
    elif len(women&tokens_stemmed) > 0:
        gender, gender_words = 'woman', women&tokens_stemmed
    elif len(men&tokens_stemmed) > 0:
        gender, gender_words = 'man', men&tokens_stemmed

    return gender, list(tokens_stemmed - gender_words)

def find_category(tokens_stemmed): #returns all categories listed in title and corresponding gender if woman only or men only category present, gender empty otherwise
    unisex_categories = set(['shirt', 't-shirt', 'sweater', 'cardigan', 'jacket', 'coat', 'blazer', 'hoodie', 'shrug', 'kurta', 'waistcoat', 'capri', 'dungree', 'pant', 'skirt', 'cargo', 'shorts', 'pyjama', 'jean'])
    women_only_categories = set(['top', 'dress', 'sare', 'lehenga-choli', 'dupatta', 'palazzo', 'bra', 'gown', 'legging', 'nighty', 'salwar suit', 'anarkali', 'blouse', 'churidar', 'jumpsuit', 'patiala'])
    men_only_categories = set(['dhoti', 'vest', 'boxer'])

    gender, category  = '', women_only_categories&tokens_stemmed

    if len(category) == 0:
        category = men_only_categories&tokens_stemmed
        if len(category) == 0:
            category = unisex_categories&tokens_stemmed
        else:
            gender, category = 'man', category | unisex_categories&tokens_stemmed
    else:
        gender, category = 'woman', category | unisex_categories&tokens_stemmed

    if gender == '' and len(category) > 0:
        gender = 'unisex'

    return gender, list(category), list(tokens_stemmed - category)

def extract_tags(product):
    corpus = make_corpus(product['title'].replace(product['brand'], '').lower().split(" "))
    tokens = tokenizer_nltk.tokenize(corpus)
    tokens_stop = delete_stop_words(tokens, stop_words)
    tokens_lemm = lemmatize(tokens_stop, lemmatizer)
    tokens_stemmed = stem(tokens_lemm, stemmer)

    gender, tokens_stemmed = find_gender(set(tokens_stemmed))
    gender_from_category, category, tokens_stemmed = find_category(set(tokens_stemmed))

    if gender == '': # use the gender from category if not explicitly mentioned in title
        gender = gender_from_category

    return tokens_stemmed, gender, category

df = pd.read_csv("sample_fk_catalog.csv.xls")
product = df.loc[4]

tokenizer_nltk = RegexpTokenizer(r'\w+(?:-\w+)*') # words with optional internal hyphen(-) included
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

product_row = {}
product_row['product_tags'], product_row['gender'], product_row['category'] = extract_tags(product)
print(product_row)
