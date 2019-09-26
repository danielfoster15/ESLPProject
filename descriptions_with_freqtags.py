import re
ids = []
descrips = []
with open('freqtags2.txt', 'r') as f:
    for line in f:
        line = line.split('\t')
        uid, sid, hashtag = line[0], line[1], line[2]
        ids.append((uid,sid))


with open('descriptions.txt') as f:
    for line in f:
        line = line.split('\t')
        if (line[0], line[1]) in ids:
            notags = re.sub(r'#\w+', '', line[2].strip('\n'))
            descrips.append((line[0], line[1], line[2].strip('\n'), notags))

with open('descriptions_with_freqtags.txt', 'w') as f:
    for uid, sid, tweet, notags in descrips:
        f.write(uid+'\t'+sid+'\t'+tweet+'\t'+notags+'\n')

