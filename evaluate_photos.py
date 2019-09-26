from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import re

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		tokens=line.split()
		uid, sid, image_desc = tokens[0], tokens[1], tokens[2:]
		image_id='uid='+str(uid)+'_sid='+str(sid)
		dataset.append(image_id)
	return dataset

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		# split line by white space
		tokens = line.split()
		# split id from description
		uid, sid, image_desc = tokens[0], tokens[1], tokens[2:]

		image_id='uid='+str(uid)+'_sid='+str(sid)
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = '< ' + ' '.join(image_desc) + ' >'
			# store
			descriptions[image_id].append(desc)
			#print(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
from generatecaption import generate_desc

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	tn = 0
	tp = 0
	fp = 0
	fn = 0
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [[re.sub('#', '', w.lower()) for w in d.split() if w not in ['>', '<', 'demsinphilly']] for d in desc_list][0]
		actual.append(references)
		yhat = [w for w in yhat.split() if w not in ['>', '<', 'demsinphilly']]
		predicted.append(yhat)
		print(references, yhat)
	vocab = [word for sublist in actual for word in sublist]
	zipped = (zip(actual, predicted))
	#print(list(zipped))
	for trues, preds in zipped:
		for pred in preds:
			if pred in trues:
				tp+=1
			elif pred not in trues:
				fp +=1
		for truth in trues:
			if truth not in preds:
				fn += 1
		for word in vocab:
			if word not in preds and word not in trues:
				tn += 1
	print(tn, tp, fn, fp)
	prec = tp / (tp+fp)
	rec = tn / (tn+fn)
	f1 = 2 * ((prec * rec) / (prec + rec))
	# calculate BLEU score

	print('Precision: %f' % prec)
	print('Recall: %f' % rec)
	print('F1: %f' % f1)

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'freqtags2.txt'
fullset=load_set(filename)
train= fullset[:9357]
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('freqtags2.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set

test= fullset[9357:]
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('freqtags2.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model-ep005-loss4.411-val_loss4.687.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)