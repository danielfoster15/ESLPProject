from random import shuffle

food_tweets = []
sports_tweets = []
political_convention_tweets = []
beauty_tweets = []

with open('descriptions_with_freqtags_test.txt') as f:
	for line in f:
		if len(line.split('\t')) == 4:
			uid, sid, tweet, notags = line.split('\t')
		elif len(line.split('\t')) == 3:
			uid, sid, tweet = line.split('\t')
			notags = None

		tweet = tweet.lower()
		print(tweet)
		if notags:
			if '#healthy' in tweet:
				food_tweets.append(notags)
			elif '#olympics' in tweet or '#teamusa' in tweet:
				sports_tweets.append(notags)
			elif '#demsinphilly' in tweet or "#dncinphl" in tweet:
				political_convention_tweets.append(notags)
			elif '#bbloggers' in tweet or '#beauty' in tweet or "#makeup" in tweet:
				beauty_tweets.append(notags)

tweet_sets = [food_tweets, sports_tweets, political_convention_tweets, beauty_tweets]

for tweet_set in tweet_sets:
	shuffle(tweet_set)

with open('Examples/food_tweets.txt', 'w') as f:
	for tweet in food_tweets[:26]:
		f.write(tweet)
with open('Examples/sports_tweets.txt', 'w') as f:
	for tweet in sports_tweets[:26]:
		f.write(tweet)
with open('Examples/political_convention_tweets.txt', 'w') as f:
	for tweet in political_convention_tweets[:26]:
		f.write(tweet)
with open('Examples/beauty_tweets.txt', 'w') as f:
	for tweet in beauty_tweets[:26]:
		f.write(tweet)