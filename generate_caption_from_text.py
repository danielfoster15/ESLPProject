from pickle import load
import numpy as np
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from gensim.models import Doc2Vec
from nltk.tokenize import TweetTokenizer

doc2vec_model = Doc2Vec.load('doc2vecmodel.h5')

tweettokenizer = TweetTokenizer()

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def sample(preds, temperature=.5):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return argmax(probas)


def get_vocab():
	tweet_vocab_lookup = {}
	id = 0
	tweettokens = []
	with open('descriptions_with_freqtags.txt', 'r') as f:
		for line in f:
			splitline = line.split('\t')
			if len(splitline) == 4:
				uid, sid, hashtweet, tweetonly = splitline
			elif len(splitline) == 3:
				uid, sid, hashtweet = splitline
				tweetonly = ''
			if tweetonly == '' or tweetonly == '\n' or tweetonly.isspace():
				tokens = [' ']
			else:
				tokens = tweettokenizer.tokenize(tweetonly)
			tweettokens.append((tokens, uid, sid))
			for word in tokens:
				if word not in tweet_vocab_lookup:
					tweet_vocab_lookup[word] = id
					id += 1
	return tweet_vocab_lookup

def load_tweet_only(tweet, tweet_vocab_lookup):
	# load all features
	tweettokens = tweettokenizer.tokenize(tweet)
	tweetvec = doc2vec_model.infer_vector(tweettokens)
	# filter features
	return tweetvec


# generate a description for an image
def generate_desc_from_tweet(model, tokenizer, tweetvec, max_length):
	# seed the generation process
	in_text = '<'
	# iterate over the whole length of the sequence
	for diversity in [.2]:
		for i in range(max_length):
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			#print('tweetvec seq', tweetvec.shape, sequence)
			# predict next word
			yhat = model.predict([[tweetvec], sequence], verbose=0)[0]
			# convert probability to integer
			yhat = sample(yhat, diversity)
			# map integer to word
			word = word_for_id(yhat, tokenizer)
			#stop if word already used
			if word in in_text.split():
				in_text += ' >'
				break
			# stop if we cannot map the word
			if word is None:
				break
			# append as input for generating the next word
			in_text += ' ' + word
			# stop if we predict the end of the sequence
			if word == '>':
				break
	return in_text+''



# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 15
# load the model
text_model = load_model('text-model-ep004-loss4.912-val_loss4.771.h5')
text_model.summary()
# load and prepare the photograph
food_tweets = 'Examples/food_tweets.txt'
sports_tweets = 'Examples/sports_tweets.txt'
political_convention_tweets = 'Examples/political_convention_tweets.txt'
beauty_tweets = 'Examples/beauty_tweets.txt'


food_preds = []
sports_preds = []
political_convention_preds = []
beauty_preds = []
for item in [food_tweets, sports_tweets, political_convention_tweets, beauty_tweets]:
	with open(item) as f:
		cat = item.split('/')[1].split('_')[0]
		tweet_vocab_lookup = get_vocab()
		for line in f:
			tweetvector = load_tweet_only(line.strip('\n'), tweet_vocab_lookup)
			prediction = generate_desc_from_tweet(text_model, tokenizer, tweetvector, max_length)
			print(prediction)
			if cat == 'food':
				food_preds.append(prediction)
			elif cat == 'sports':
				sports_preds.append(prediction)
			elif cat == 'beauty':
				beauty_preds.append(prediction)
			elif cat == 'political':
				political_convention_preds.append(prediction)
		with open('Examples/'+cat+'_predictions.txt', 'w') as f:
			if cat == 'food':
				for hashtag in food_preds:
					f.write(hashtag+'\n')
			elif cat == 'sports':
				for hashtag in sports_preds:
					f.write(hashtag+'\n')
			elif cat == 'beauty':
				for hashtag in beauty_preds:
					f.write(hashtag+'\n')
			elif cat == 'political':
				for hashtag in political_convention_preds:
					f.write(hashtag+'\n')