import re
hashtweets=[]

def findandwritehashtags(filename, writefilename):
#returns only tweets that contain hashtags and then removes all text except hashtags writing to a new file of jpg name and hashtags per file
	with open(filename) as f:
		for line in f:
			uid, sid=line.split()[:2]
			tweet=' '.join(line.split()[2:])
			if '#' in tweet:
				hashtweets.append((uid, sid, tweet))
	f.close()
	hashtags=[]
	for uid, sid, tweet in hashtweets:
		hashtagsonly=re.findall(r'\B(\#[a-zA-Z]+\b)(?!;)', tweet)
		if len(hashtagsonly) > 0:
			hashtags.append((uid, sid, hashtagsonly))

	with open(, 'w') as f:
		for uid, sid, tweet in hashtags:
			f.write(uid+'\t'+sid+'\t'+' '.join(tweet)+'\n')
	f.close()

if __name__ == '__main__':
	findandwritehashtags('/srv/devel/foster/Desktop/ESLP/twitter100k/twitter100K/descriptions.txt', 'hashtags.txt')