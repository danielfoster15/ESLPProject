
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk.tokenize import TweetTokenizer



tweettokenizer = TweetTokenizer()

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
	return model

filename = 'freqtags2.txt'
fullset=load_set(filename)
train = fullset
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('freqtags2.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
doc2vec_model = doc_to_vec('descriptions_with_freqtags.txt', train)

doc2vec_model.save('doc2vecmodel.h5')

