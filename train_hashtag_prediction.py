from numpy import array
from os import listdir
from pickle import load
from random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import TweetTokenizer
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


doc2vec_model = Doc2Vec.load('doc2vecmodel.h5')


tweettokenizer = TweetTokenizer()

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

# load photo features
def load_tweet_only(txtfile, dataset):
	# load all features
	tweet_vocab_lookup = {}
	id = 0
	tweettokens = []
	tweetvecs = {}
	with open(txtfile, 'r') as f:
		for line in f:
			splitline = line.split('\t')
			if len(splitline) == 4:
				uid, sid, hashtweet, tweetonly = splitline
			elif len(splitline)==3:
				uid, sid, hashtweet = splitline
				tweetonly= ''
			else:
				print('no!')
			if tweetonly == '' or tweetonly == '\n' or tweetonly.isspace():
				tokens = [' ']
			else:
				tokens = tweettokenizer.tokenize(tweetonly)
			tweettokens.append((tokens, uid, sid))
			for word in tokens:
				if word not in tweet_vocab_lookup:
					tweet_vocab_lookup[word] = id
					id += 1
			if tokens == [' ']:
				print('empty id thing:      ', tweet_vocab_lookup[' '])
	for tweet, uid, sid in tweettokens:
		tweetvec = doc2vec_model.infer_vector(tweet).reshape((1,150))
		identifier = 'uid='+str(uid)+'_sid='+str(sid)
		if identifier in dataset:
		#print(identifier)
			tweetvecs[identifier] = tweetvec
	# filter features
	return tweetvecs

def doc_to_vec(txtfile, dataset):
	labeled_tweets = []
	tweettokens = []
	with open(txtfile, 'r') as f:
		for line in f:
			splitline = line.split('\t')
			if len(splitline) == 4:
				uid, sid, hashtweet, tweetonly = splitline
			elif len(splitline)==3:
				uid, sid, hashtweet = splitline
				tweetonly= ''
			else:
				print('no!')
			if tweetonly == '' or tweetonly == '\n' or tweetonly.isspace():
				tokens = [' ']
			else:
				tokens = tweettokenizer.tokenize(tweetonly)
			tweettokens.append((tokens, uid, sid))
	for tweet, uid, sid in tweettokens:
		identifier = 'uid='+str(uid)+'_sid='+str(sid)
		if identifier in dataset:
			labeled_tweets.append(TaggedDocument(tweet, identifier))
	model = Doc2Vec(dm=1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
	model.build_vocab(labeled_tweets)
	for epoch in range(20):
		model.train(labeled_tweets, epochs=model.iter, total_examples=model.corpus_count)
		print("Epoch #{} is complete.".format(epoch + 1))
	# filter features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer(filters = '!"%&()*+,-./:;=?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		print(desc_list)
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				print('tweet vector:           ', photos[key].shape)
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(150,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# train dataset

# load training dataset (6K)
filename = 'freqtags2.txt'
fullset=load_set(filename)
train = fullset[:9000]
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('freqtags2.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_tweet_only('descriptions_with_freqtags.txt', train)
#print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset

# load test set
filename = 'freqtags2.txt'
test= fullset[9357:]
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('freqtags2.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_tweet_only('descriptions_with_freqtags.txt', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# fit model

# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'text-model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=50, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))