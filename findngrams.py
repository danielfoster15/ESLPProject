import re
from nltk.util import ngrams
import nltk
hashtweets=[]

def findngrams(filename, n):
#returns only tweets that contain hashtags and then removes all text except hashtags writing to a new file of jpg name and hashtags per file
	with open(filename) as f:
		for line in f:
			uid, sid=line.split()[:2]
			tweet=' '.join(line.split()[2:])
			if '#' in tweet:
				hashtweets.append((uid, sid, tweet))
	f.close()
	hashtags=[]
	ngramdict={}
	for uid, sid, tweet in hashtweets:
		hashtagsonly=re.findall(r'\B(\#[a-zA-Z]+\b)(?!;)', tweet)
		hashtagslower=[x.lower() for x in hashtagsonly]
		if len(hashtagsonly) > 0:
			hashtags.append((uid, sid, hashtagsonly, hashtagslower))

	for image in hashtags:
		for hashtag in image[3]:
	#find most frequent ngrams as defined in the function with n
			for gram in list(ngrams(image[3], n)):
				if gram not in ngramdict:
					ngramdict[gram] = 1
				else:
					ngramdict[gram] +=1
	fdist=nltk.FreqDist(ngramdict)
	freqtags=[result[0][0] for result in fdist.most_common(400)[2:]]
	mostfreqhashtags=[]
	for uid, sid, tweet, tweetlower in hashtags:
		if bool(set(tweetlower) & set(freqtags)):
			mostfreqhashtags.append((uid, sid, tweet))
	with open('/srv/devel/foster/Desktop/ESLP/twitter100k/twitter100K/freqtags2.txt', 'w') as f:
		for uid, sid, tweet in mostfreqhashtags:
			f.write(uid+'\t'+sid+'\t'+' '.join(tweet)+'\n')
	f.close()





if __name__ == '__main__':
	totaltweets=0
	findngrams('/srv/devel/foster/Desktop/ESLP/twitter100k/twitter100K/descriptions.txt', 1)
