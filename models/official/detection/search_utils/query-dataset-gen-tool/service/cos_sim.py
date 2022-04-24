import math
from urllib.parse import urlparse
from urllib.parse import urlunparse
import pdb
import pandas as pd

def parseUrl(url):
	comp = urlparse(str(url))
	path = comp.path
	path = path.replace("{@width}", "500")
	path = path.replace("{@height}", "300")
	query = comp.query
	query = query.replace("{@quality}", "100")
	url = urlunparse([comp.scheme, comp.netloc, path, comp.params, query, comp.fragment])
	return url

def get_prods(data):
	prods=[]
	for no, row in data.iterrows():
		prods.append(row)
	return prods

def get_weight(idx):
	if idx == 0:
		return 5
	elif idx == 1:
		return 4
	return 1

def get_weight_for_query_word(query_candidate_words):
	# garments specific for single category has higher weight compared to garments common to all categories.
	# for example: saree in female, vest in male

	gender_categories = ['sare', 'patiala', 'dupatta', 'skirt', 'nighty', 'anarkali', 'blouse', 'churidar', 'bra', 'dhoti', 'vest', 'boxer']
	for gender_category in gender_categories:
		if gender_category in query_candidate_words:
			return 4

	categories = ['shirt', 'top', 'tshirt', 'sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'hair accessory', 'tights', 'stockings']
	for category in categories:
		if category in query_candidate_words:
			return 3
    
	colors = ['red', 'blue', 'yellow' 'green', 'black', 'white', 'grey', 'brown', 'cream', 'purple', 'orange', 'pink', 'silver']
	for color in colors:
		if color in query_candidate_words:
			return 2
    
	return 1

def get_union(lst1, lst2):
	union = []
	for val in lst1:
		union.append(val)
	for val in lst2:
		if val in union:
			continue
		union.append(val)
	return union

def get_scores(data, query):
	scores = []
	cols = list(data["tags"])
	for col in cols:
		num = 0
		idx = 0
		qden = 0
		dden = 0
		union = get_union(query, col)
		score = 0
		for val in union:
			w = get_weight(idx)
			a = 0
			b = 0
			if val in query:
				a=1
			if val in col:
				b=1
			num = num + (w*a*b)
			qden = qden + (w*a*w*a)
			dden = dden + (b*b)
			idx = idx + 1
		if num > 0:
			score = num/math.sqrt(qden)
			score = score / math.sqrt(dden)

	data["scores"] = scores

def get_normal_scores(data, query):
	query = inverse_map_query(query)
	scores = []
	match_vals = []
	cols = list(data["tags"])

	for inx, col in enumerate(cols):
		score = 0
		match_vals.append([query])
		match_vals[inx].append(col)

		for query_candidate_words in query:
			weight = get_weight_for_query_word(query_candidate_words)
			score = score + weight*find_score(query_candidate_words, col)[0]

			match_vals[inx].append([query_candidate_words, score, find_score(query_candidate_words, col)[1]])

		scores.append(score)

	data["scores"] = scores
	data["match_vals"] = match_vals

def find_score(query_candidate_words, col):
	for candidate in query_candidate_words:
		if candidate in col:
			candidate_pass = [candidate, candidate in col]
			return [1, candidate_pass]

	return [0, '']

def sort_func(prod):
	return -1 * prod.scores

def match_items(data, query_tags):
	get_normal_scores(data, query_tags)
	prods = get_prods(data)
	prods.sort(key=sort_func)
	for prod in prods:
		prod.image_url = parseUrl(prod.image_url)

# 	pretty_print_prods(prods[:5])
	return prods[:5]

def inverse_dict(file_name=["search_utils/query-dataset-gen-tool/temp/inverse.csv", "search_utils/query-dataset-gen-tool/temp/inverse_attribute_values.csv"]):
	data = pd.read_csv(file_name[1])
	root_words, words = data["Root"], data["words"]
	d1 = {}
	for r,w in zip(root_words, words):
		w = w.replace("[", '')
		w = w.replace("]", '')
		w = w.replace("'", '')
		w = w.replace('"', '')
		w = w.replace(" ", '')
		w = w.split(',')
		d1.update({r: w})
    
	data = pd.read_csv(file_name[0])
	root_words, words = data["Root"], data["words"]
	inverse_dict = {}
	for r,w in zip(root_words, words):
		w = w.replace("[", '')
		w = w.replace("]", '')
		w = w.replace("'", '')
		w = w.replace('"', '')
		w = w.replace(" ", '')
		w = w.split(',')
		for each_w in w:
			inverse_dict.update({each_w: [r]})
			if each_w in d1:
				inverse_dict[each_w] += d1[each_w]
			if r in d1:
				inverse_dict[each_w] += d1[r]

	return inverse_dict

def inverse_map_query(query):
	inverse_dictionary = inverse_dict()

	inverse_query = []
	for query_word in query:
		if query_word in inverse_dictionary.keys():
			inverse_query.append(inverse_dictionary[query_word])
		else:
			inverse_query.append([query_word])

	return inverse_query

def pretty_print_prods(prods):
	for prod in prods:
		for val in prod.match_vals:
			print(val)
		print(prod.scores, '\n')
