import os
from pickle import load
import numpy as np
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from train_hashtag_prediction import load_tweet_only

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
def get_tweet_vecs(filename):
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
			if tweetonly == '':
				tokens = [' ']
			else:
				tokens = tweettokenizer.tokenize(tweetonly)
			print(tokens,uid,sid)
			tweettokens.append((tokens, uid, sid))
			for word in tokens:
				if word not in tweet_vocab_lookup:
					tweet_vocab_lookup[word] = id
					id += 1
	for tweet, uid, sid in tweettokens:
		tweetvec = [tweet_vocab_lookup[word] for word in tweet]
		identifier = 'uid='+str(uid)+'_sid='+str(sid)
		#print(identifier)
		tweetvecs[identifier] = tweetvec

def sample(preds, temperature=.5):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return argmax(probas)

# generate a description for an image
def generate_desc_from_tweet(model, tokenizer, tweet, max_length):
	# seed the generation process
	in_text = '<'
	# iterate over the whole length of the sequence
	for diversity in [.2]:
		for i in range(max_length):
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			# predict next word
			yhat = model.predict([tweet,sequence], verbose=0)[0]
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
model = load_model('model-ep005-loss4.411-val_loss4.687.h5')
# load and prepare the photograph
food_pics = ['Examples/food/'+f for f in os.listdir('Examples/food/')]
sports_pics = ['Examples/sports/' + f for f in os.listdir('Examples/sports/')]
beauty_pics = ['Examples/beauty/' + f for f in os.listdir('Examples/beauty/')]
political_convention = ['Examples/political_convention/' + f for f in os.listdir('Examples/political_convention/')]
for filename in sports_pics:
	if 'hashtag' not in filename:
		writefile = filename.split('.')[0]+'_hashtag.txt'
		if writefile.split('/')[2] not in os.listdir('Examples/sports/'):
			with open(writefile, 'w') as f:
				print(filename)
				tweetvecs = load_tweet_only()
		# generate description
				description = generate_desc_from_tweet(model, tokenizer, tweet, max_length)
				print(description)
				f.write(description)