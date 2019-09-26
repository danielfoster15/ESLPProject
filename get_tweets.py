import re
hashtweets=[]

def findandwritetweets(filename, writefilename):
#returns only tweets that contain hashtags and then removes all text except hashtags writing to a new file of jpg name and hashtags per file
	with open(filename) as f:
		for line in f:
			uid, sid=line.split()[:2]
			tweet=' '.join(line.split()[2:])
			if '#' in tweet:
				hashtweets.append((uid, sid, tweet))
	f.close()
	tweets=[]
	for uid, sid, tweet in hashtweets:
		hashtagsonly=re.findall(r'\B(\#[a-zA-Z]+\b)(?!;)', tweet)
		if len(hashtagsonly) > 0:
			nothashtagsonly = re.sub(r'\B(\#[a-zA-Z]+\b)(?!;)', '', tweet)
			tweets.append((uid, sid, nothashtagsonly))


	with open(writefilename, 'w') as f:
			for uid, sid, tweet in tweets:
				f.write(uid+'\t'+sid+'\t'+tweet+'\n')

if __name__ == '__main__':
	findandwritetweets('descriptions.txt', 'tweets.txt')