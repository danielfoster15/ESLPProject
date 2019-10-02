# ESLPProject

The main python scripts in this repo do this:

1. preparephotodata.py -- uses VGG16 model to save features file to the directory as 'features.pkl' on a set of photos  
   this one takes 2 hours~ if you don't have a GPU, on a GPU 10 minutes
    
2. preparetextdata.py -- creates a file called "descriptions" with photo id and tweet

3. scripts for finding frequent hashtags and their tweets - findhashtags.py, descriptions_with_freqtags.py, freq_tweets_only.py, get_tweets.py

4.  tokenizer.py -- creates a vocabulary of hashtags for the model to use saved as tokenizer.pkl

5a. trainmodel.py -- using 'freqtags2.txt' (frequent hashtags), tokenizer.pkl, and features.pkl trains the image-based model

5b. train_hashtag_prediction.py -- trains the text-based model like 5a, but uses doc2vec model for tweets

6. doc_to_vec.py -- trains a doc2vec model for the tweets

7. generatecaption.py -- generates caption from image model given an image, generate_caption_from_text.py -- generates caption from text model given a sentence
