
# coding: utf-8

# In[2]:


import re
import sys
from nltk.stem.porter import PorterStemmer
import pandas as pd
import urllib
import luigi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import operator
import pickle
from boto.s3.connection import S3Connection
from boto.s3.key import Key


# In[ ]:


class download_csv(luigi.Task):
    
    def run(self):
        urllib.request.urlretrieve("https://s3.amazonaws.com/finalprojectads/training_data.csv", filename= 'training_data.csv')
        df = pd.read_csv("training_data.csv", encoding='ISO-8859-1', header=None)
        df.to_csv(self.output().path,index = False)

    def output(self):
        return luigi.LocalTarget("training_data.csv")


# In[ ]:


class clean_data(luigi.Task):

	def requires(self):
		yield download_csv()
    
	def run(self):
		data = pd.read_csv(download_csv().output().path, encoding='ISO-8859-1', header=0)
		data = data.rename(columns={'0':'Target','1':'TweetId', '2':'DateTime', '3':'Flag','4':'User','5':'Tweet'})
		only_text = pd.DataFrame(data['Tweet'])
        
		def handle_emojis(tweet):
			# Smile -- :), : ), :-), (:, ( :, (-:, :')
			tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
			# Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
			tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
			# Love -- <3, :*
			tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
			# Wink -- ;-), ;), ;-D, ;D, (;,  (-;
			tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
			# Sad -- :-(, : (, :(, ):, )-:
			tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
			# Cry -- :,(, :'(, :"(
			tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
			return tweet
		def preprocess_word(word):
			# Remove punctuation
			word = word.strip('\'"?!,.():;')
			# Convert more than 2 letter repetitions to 2 letter
			# funnnnny --> funny
			word = re.sub(r'(.)\1+', r'\1\1', word)
			# Remove - & '
			word = re.sub(r'(-|\')', '', word)
			return word
		def is_valid_word(word):
			# Check if word begins with an alphabet
			return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)
		def preprocess_tweet(tweet):
			processed_tweet = []
			#Lower all the words in tweet
			tweet = tweet.lower()
			#Replace all the URL's with word URL 
			tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
			# Replace @handle with the word USER_MENTION
			tweet = re.sub(r'@[\S]+', '', tweet)
			# Replaces #hashtag with hashtag
			tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
			# Remove RT (retweet)
			tweet = re.sub(r'\brt\b', '', tweet)
			# Replace 2+ dots with space
			tweet = re.sub(r'\.{2,}', ' ', tweet)
			# Strip space, " and ' from tweet
			tweet = tweet.strip(' "\'')
			# Replace emojis with either EMO_POS or EMO_NEG
			tweet = handle_emojis(tweet)
			# Replace multiple spaces with a single space
			tweet = re.sub(r'\s+', ' ', tweet)
			words = tweet.split()

			for word in words:
				word = preprocess_word(word)
				if is_valid_word(word):
					processed_tweet.append(word)


			return ' '.join(processed_tweet)        
			
		target = []
		processed_tweet_set = []
		for i in range(0, 1600000):
			target.append(0 if data['Target'][i] == 0 else 1)
			processed_tweet = preprocess_tweet(only_text['Tweet'][i])
			processed_tweet_set.append(processed_tweet)
		
		processed_tweets = pd.DataFrame(pd.Series(processed_tweet_set), columns={'Tweet'})
		
		data['Processed_Tweets'] = processed_tweets
		cleaned_data = data[['Target','TweetId','DateTime','Flag','User','Processed_Tweets']]
		cleaned_data.to_csv(self.output().path,index=False)
            
	def output(self):
		return luigi.LocalTarget("cleaneddata.csv")


# In[ ]:


class train_model(luigi.Task):
	def requires(self):
		yield clean_data()
	def run(self):
		cleaned_data = pd.read_csv(clean_data().output().path, encoding='ISO-8859-1', header=0)
		cleaned_data.dropna(inplace=True)
		processed_tweet_set = cleaned_data.Processed_Tweets.tolist()
		target = cleaned_data.Target.apply(lambda x:1 if x>0 else 0).tolist()
		X_train, X_test, y_train, y_test = train_test_split(processed_tweet_set,target, test_size=0.2, random_state=42)
		model=LogisticRegression(C=1.)

		tfv=TfidfVectorizer(min_df=0, max_features=None, strip_accents='unicode',lowercase =True,
							analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
							use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")

		transformed_data=tfv.fit_transform(X_train)
		pickle.dump(tfv, open('tfv-model.sav','wb'))

		#Logistic Regression

		filename = 'initial-model.sav'
		model.fit(transformed_data,y_train)
		#Finding the probability of the tweets to be positive
		probability_to_be_positive=model.predict_proba(transformed_data)[:,1]

		#ROC Score of the predictions
		model_performance = pd.DataFrame(pd.Series(['AUC']), columns={'Parameter'})
		model_performance['Score'] = roc_auc_score(y_train,probability_to_be_positive)
		model_performance.to_csv('model_summary.csv')
		#print('auc score: ',roc_auc_score(y_train,probability_to_be_positive))

		pickle.dump(model, open(filename, 'wb'))
		my_files = []
		my_files.append('tfv-model.sav')
		my_files.append('initial-model.sav')
		my_files.append('model_summary.csv')
		model_file_names = pd.DataFrame(my_files)
		model_file_names.to_csv(self.output().path,index = False)
		
	def output(self):
		return luigi.LocalTarget("model_file_names.csv")

class upload2S3(luigi.Task):
	accessKey = luigi.Parameter()
	secretAccessKey = luigi.Parameter()
	def requires(self):
		yield train_model()
	def run(self):
		# Get Access keys from command line 
		accessKey = self.accessKey
		secretAccessKey = self.secretAccessKey
		try:
			#Creating S3 Connection using Access and Secrect access key
			print('Starting S3 Connection')
			conn = S3Connection(accessKey, secretAccessKey)
			print('Connection Successful')
			# Connecting to specified bucket
			print('connecting to bucket')
			b = conn.get_bucket('finalprojectads')
			print('connecting to Successful')
			#Initializing Key
			k = Key(b)
			#Uploading pickle and model performance files to S3 Bucket
			print('Starting to upload')
			onlyfiles = pd.read_csv(train_model().output().path)['0'].tolist()
			for i in onlyfiles:
				k.key = i
				k.set_contents_from_filename(i)
				k.set_acl('public-read')
			print('Upload Completed')
		except:
			print("Amazon credentials or location is invalid")
if __name__=='__main__':
	luigi.run()

