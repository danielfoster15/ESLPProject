import re
sids = []
descrips = []
with open('freqtags2.txt', 'r') as f:
    for line in f:
        line = line.split('\t')
        uid, sid, hashtag = line[0], line[1], line[2]
        sids.append(sid)


with open('tweets.txt') as f:
    for line in f:
        uid, sid, tweet = line.split('\t')
		descrips.append(tweet)

with open('freqtags_tweets_only.txt', 'w') as f:
    for uid, sid, tweet, notags in descrips:
        f.write(uid+'\t'+sid+'\t'+tweet+'\t'+notags+'\n')

