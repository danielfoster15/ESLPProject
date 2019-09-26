from keras.preprocessing.text import Tokenizer
from pickle import dump
from os import listdir

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
	return descriptions


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in ' '.join(descriptions[key]).split()]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	print(lines)
	tokenizer = Tokenizer(filters = '!"%&()*+,-./:;=?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load training dataset (6K)
filename = '/srv/devel/foster/Desktop/ESLP/twitter100k/twitter100K/freqtags2.txt'
train = load_set(filename)[:9000]
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions(filename, train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))