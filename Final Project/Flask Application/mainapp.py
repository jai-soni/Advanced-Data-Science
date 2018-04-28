from flask import Flask, render_template, redirect, url_for, request,jsonify
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from os import listdir
from os.path import isfile, join
from pandas.io.json import json_normalize
import urllib
import sys
import nltk
import numpy as np
import tweepy
import aylien_news_api
from aylien_news_api.rest import ApiException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from geopy.geocoders import Nominatim 
from textblob import TextBlob
import re
import pickle
from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
#import ray 

#Initializing Ray, and providing number of CPU's to use
#ray.init(num_cpus=10, redirect_output=True)

nltk.download('brown')
nltk.download('punkt')

# create an instance of the API class

from bokeh.plotting import *
from numpy import pi


def createPlots(data,i):
	# define starts/ends for wedges from percentages of a circle
	percents = data
	starts = [p*2*pi for p in percents[:-1]]
	ends = [p*2*pi for p in percents[1:]]

	# a color for each pie piece
	colors = ["red", "green", "blue", "orange", "yellow"]

	p = figure(x_range=(-1,1), y_range=(-1,1))

	p.wedge(x=0, y=0, radius=1, start_angle=starts, end_angle=ends, color=colors)

	# display/save everythin  
	output_file("pie.html")



def get_aylien_news(keyword,app_id,app_key):
    
    aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = app_id
    aylien_news_api.configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = app_key
    
    api_instance = aylien_news_api.DefaultApi()

    opts = {
      'title': keyword,
      'sort_by': 'social_shares_count.facebook',
      'language': ['en'],
      'not_language': ['es', 'it'],
      'published_at_start': 'NOW-2DAYS',
      'published_at_end': 'NOW-1DAYS'
    }

    try:
        # List stories
        api_response = api_instance.list_stories(**opts)
        """
        print("API called successfully. Returned data: ")
        print("========================================")
        for story in api_response.stories:
            print(story.title + " / " + story.source.name)
            """
    except ApiException as e:
        print("Exception when calling DefaultApi->list_stories: %sn" % e)
        return None
    return api_response


# Get Search Query for twitter 
def get_query(title):
    blob = TextBlob(title)
    myQuery = ' AND '.join(blob.noun_phrases)
    #print("search query will be = ",myQuery)
    return myQuery


# Functions to clean the tweets

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' good ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' good ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' good ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' good ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' bad ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' bad ', tweet)
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


#@ray.remote
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

def loc_clean(x):
    if x =='':
        return 'Not Avaiable'
    else:
        return x
    
# Functions to get vader sentiments

def get_vader_sentiment(x):
    if(x>0.05):
        return 'Positive'
    elif (x<-0.05):
        return 'Negative'
    else:
        return 'Neutral'

def get_vader(all_tweets):
    analyzer = SentimentIntensityAnalyzer()
    vs_compound = []
    for tweet in all_tweets.tweets.tolist():
        vs_compound.append(analyzer.polarity_scores(tweet)['compound'])
    all_tweets['vader_sentiment'] = pd.Series(vs_compound).apply(get_vader_sentiment)
    return all_tweets

# Get Coordinates of DataFrame

"""#@ray.remote
def get_geo(geolocator,x):
    try:
        loc = geolocator.geocode(x)
        if loc:
            return loc
        else:
            return np.nan
    except:
        pass
    return np.nan

# Get Coordinates of DataFrame
def get_coordinates(df):
    geolocator = Nominatim()

    # Go through all tweets and add locations to 'coordinates' dictionary
    coordinates = {'latitude': [], 'longitude': []}  
    list_loc=[]
    for count, user_loc in enumerate(df.location):  
    #location = geolocator.geocode(user_loc)
        list_loc.append(get_geo(geolocator,user_loc))
    #list_loc = ray.get(list_loc)        
    # If coordinates are found for location
    for i in range(0,len(list_loc)):
        if type(list_loc[i])==type(0.0):
            #print(list_loc[i])
            coordinates['latitude'].append(list_loc[i])
            coordinates['longitude'].append(list_loc[i])
        else:
            #print(list_loc[i])
            coordinates['latitude'].append(list_loc[i].latitude)
            coordinates['longitude'].append(list_loc[i].longitude)

    df['latitude'] = pd.DataFrame(coordinates)['latitude']
    df['longitude'] = pd.DataFrame(coordinates)['longitude']
    return df"""

def authenticate_tweepy(consumer_key,consumer_secret,access_token,access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth) 
    return api

def get_tweets(query,title,api):
    results = api.search(
       lang="en",
       q=query + " -rt",
       count=50,
       result_type="recent"
    )
    
    #print(len(results))
    #n_initial = len(results)
    
    # tweets Dataframe
    tweets_data = pd.DataFrame() 
    tweets = []
    pro_tweets = []
    location = []
    
    for result in results:
        if title in result.text:
            
            tweets.append(result.text)
            #result.text = preprocess_tweet.remote(result.text.replace(title,''))
            #print('Result.Text : ',result.text)
            pro_tweets.append(preprocess_tweet(result.text.replace(title,'')))
            location.append(result.user.location)
        else :
            tweets.append(result.text)
            pro_tweets.append(preprocess_tweet(result.text))
            location.append(result.user.location)
    # Change unavialble location to NA 
    location = pd.Series(location).apply(loc_clean).tolist()
    #pro_tweets = ray.get(pro_tweets)
    # Adding tweets and processed tweets to dataframe
    
    tweets_data.insert(loc=0, column='tweets', value=pd.Series(tweets))
    tweets_data.insert(loc=1, column='pro_tweets', value=pd.Series(pro_tweets))
    tweets_data.insert(loc=2, column='location', value=pd.Series(location))
    #Removing empty tweets and most occuring exact same tweet
    
    #tweets_data = tweets_data.replace('',np.nan)
    #print('Inside get_tweets, len of tweets_data',len(tweets_data))
    if len(tweets_data)>1:
        tweets_data = tweets_data.replace('',np.nan)
        if (len(tweets_data['pro_tweets'].value_counts())>2):
            tweets_data = tweets_data.replace(tweets_data['pro_tweets'].value_counts()[:2].index.tolist(),np.nan)
        tweets_data.dropna(inplace=True)
        tweets_data=tweets_data.reset_index().drop('index',axis=1)
        tweets_data['article_name'] = title
        tweets_data['twitter_search'] = query
    
    return tweets_data

# Functions to get vader sentiments

def get_vader_sentiment(x):
    if(x>0.05):
        return 'Positive'
    elif (x<-0.05):
        return 'Negative'
    else:
        return 'Neutral'

def get_vader(all_tweets):
    analyzer = SentimentIntensityAnalyzer()
    vs_compound = []
    for tweet in all_tweets.tweets.tolist():
        vs_compound.append(analyzer.polarity_scores(tweet)['compound'])
    all_tweets['vader_sentiment'] = pd.Series(vs_compound).apply(get_vader_sentiment)
    return all_tweets

# Get Coordinates of DataFrame
def get_coordinates(df):
    geolocator = Nominatim()

    # Go through all tweets and add locations to 'coordinates' dictionary
    coordinates = {'latitude': [], 'longitude': []}  
    for count, user_loc in enumerate(df.location):  
        try:
            location = geolocator.geocode(user_loc)

            # If coordinates are found for location
            if location:
                coordinates['latitude'].append(location.latitude)
                coordinates['longitude'].append(location.longitude)
            else:
                coordinates['latitude'].append(np.nan)
                coordinates['longitude'].append(np.nan)
        # If too many connection requests
        except:
            pass
    df['latitude'] = pd.DataFrame(coordinates)['latitude']
    df['longitude'] = pd.DataFrame(coordinates)['longitude']
    return df

app = Flask(__name__)





@app.route("/welcome")
def welcome():
	return render_template('welcome.html')

@app.route("/checkInput", methods=['POST'])
def checkInput(chartID = 'chart_ID', chart_type = 'bar', chart_height = 350):
    error = None
    input = request.form['user_keyword']
    text = input.replace(' ','')
    if len(text) == 0:
        error = "Please enter something to be searched"
        return render_template("welcome.html", error=error)
    else:
        error=None
        keyword = input

        # Aylien Keys
        app_id = '5861e320'
        app_key = 'f6c7d9ae180951898bf6263d64dac247'

        # Tweepy keys
        consumer_key = "1Cvu0r3U1bafJE0prSuJXUhS6"
        consumer_secret = "l2NhimsNqLQbM490spvEt2sDHJs4KJEeWQ2FkHP2P0LpjggtjY"
        access_token = "420232136-pNrWEakqUOq8kbHPsX2jCTrm7ocyAyIYpHe3mCNZ"
        access_token_secret = "wfzXzbtqYfhIhsD2egUJC8l7Gy2fMcrKrw2Pq50BW9qZA"

        news_response = get_aylien_news(keyword,app_id,app_key)
        api = authenticate_tweepy(consumer_key,consumer_secret,access_token,access_token_secret)


        if(len(news_response.stories)>0):
            all_tweets = pd.DataFrame()
            for story in news_response.stories:
                query = get_query(story.title)
                #print('number : ',i,' query for: ',story.title,' is: ',query)
                story_tweets = get_tweets(query,story.title,api)
                #print('Got story_tweet : ',i,' with num of tweets :',len(story_tweets))
                all_tweets = all_tweets.append(story_tweets,ignore_index=False)
                #print('Appending : ',i,' with num of tweets :',len(story_tweets),' with total count: ',len(all_tweets))
                #print(all_tweets)
            all_tweets = all_tweets.reset_index()
            all_tweets = all_tweets.drop('index', inplace= False, axis = 1)
            all_tweets = get_vader(all_tweets)



            #Get Geo Coordinates
            #all_tweets = get_coordinates(all_tweets)


            urllib.request.urlretrieve("https://s3.amazonaws.com/finalprojectads/initial-model.sav", filename= 'initial-model.sav')

            urllib.request.urlretrieve("https://s3.amazonaws.com/finalprojectads/tfv-model.sav", filename= 'tfv-model.sav')

            tfv = pickle.load(open('tfv-model.sav','rb'))

            model = pickle.load(open("initial-model.sav", "rb"))

            print(len(all_tweets))

            processed_tweet_set = all_tweets['pro_tweets'].tolist()
            transformed_data = tfv.transform(processed_tweet_set)

            #Finding the probability of the tweets to be positive
            probability_to_be_positive=model.predict_proba(transformed_data)[:,1]
            probability_to_be_positive = probability_to_be_positive.tolist()
            for i in range(0, len(probability_to_be_positive)):
                if probability_to_be_positive[i] >= 0.6:
                    probability_to_be_positive[i] = 'positive'
                elif probability_to_be_positive[i] <= 0.25:
                    probability_to_be_positive[i] = 'negative'
                else:
                    probability_to_be_positive[i] = 'neutral'

            pd.set_option('display.max_colwidth', -1)

            all_tweets['model_sentiment'] = probability_to_be_positive

            summary = pd.DataFrame()
            articles=[]
            total_positive=[]
            total_negative=[]
            total_neutral=[]
            total_tweets =[]
            for article in all_tweets.article_name.unique().tolist():
                articles.append(article)
                total_negative.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['vader_sentiment']=='Negative']))
                total_positive.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['vader_sentiment']=='Positive']))
                total_neutral.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['vader_sentiment']=='Neutral']))
                total_tweets.append(len(all_tweets[all_tweets['article_name']==article]))
            summary.insert(loc=0, column='name', value=pd.Series(articles))
            summary.insert(loc=1, column='total_negative', value=pd.Series(total_negative))
            summary.insert(loc=2, column='total_positive', value=pd.Series(total_positive))
            summary.insert(loc=3, column='total_neutral', value=pd.Series(total_neutral))
            summary.insert(loc=4, column='total_tweets', value=pd.Series(total_tweets))
            summary.dropna(inplace=True)
            summary.reset_index(inplace=True)
            summary.drop('index',axis=1,inplace=True)
            summary.dropna(inplace=True)
            summary['total_negative'] = (summary['total_negative']*100)/summary['total_tweets']
            summary['total_positive'] = (summary['total_positive']*100)/summary['total_tweets']
            summary['total_neutral'] = (summary['total_neutral']*100)/summary['total_tweets']
            summary.drop('total_tweets',axis=1,inplace=True)
            vader_dict = summary.T.to_dict()

            #Getting summary for my model
            my_summary = pd.DataFrame()
            articles=[]
            total_positive=[]
            total_negative=[]
            total_neutral=[]
            total_tweets =[]
            for article in all_tweets.article_name.unique().tolist():
                articles.append(article)
                total_negative.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['model_sentiment']=='negative']))
                total_positive.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['model_sentiment']=='positive']))
                total_neutral.append(len(all_tweets[all_tweets['article_name']==article][all_tweets['model_sentiment']=='neutral']))
                total_tweets.append(len(all_tweets[all_tweets['article_name']==article]))
            my_summary.insert(loc=0, column='name', value=pd.Series(articles))
            my_summary.insert(loc=1, column='total_negative', value=pd.Series(total_negative))
            my_summary.insert(loc=2, column='total_positive', value=pd.Series(total_positive))
            my_summary.insert(loc=3, column='total_neutral', value=pd.Series(total_neutral))
            my_summary.insert(loc=4, column='total_tweets', value=pd.Series(total_tweets))
            my_summary.dropna(inplace=True)
            my_summary.reset_index(inplace=True)
            my_summary.drop('index',axis=1,inplace=True)
            my_summary.dropna(inplace=True)
            my_summary['total_negative'] = (my_summary['total_negative']*100)/my_summary['total_tweets']
            my_summary['total_positive'] = (my_summary['total_positive']*100)/my_summary['total_tweets']
            my_summary['total_neutral'] = (my_summary['total_neutral']*100)/my_summary['total_tweets']
            my_summary.drop('total_tweets',axis=1,inplace=True)
            my_dict = my_summary.T.to_dict()

            chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}

            positive = len(all_tweets[all_tweets['model_sentiment'] == 'positive'])
            negative = len(all_tweets[all_tweets['model_sentiment'] == 'negative'])
            neutral = len(all_tweets[all_tweets['model_sentiment'] == 'neutral'])

            series = [{"name": 'positive', "data": [positive]}, {"name": 'neutral', "data": [neutral]}, {"name":"negative", "data":[negative]}]
            title = {"text": 'Overall Sentiments'}
            xAxis = {"categories": ['Sentiment']}
            yAxis = {"title": {"text": 'No of Tweets'}}


            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(
                    background_color='white',
                    stopwords= stopwords,
                    max_words=200,
                    max_font_size=40, 
                    scale=3,
                    random_state=1 # chosen at random by flipping a coin; it was heads
                ).generate(str(processed_tweet_set))

            fig = plt.figure(1, figsize=(12, 6))
            plt.axis('off')
            plt.imshow(wordcloud)
            fig.savefig('static/test.jpg')
            df = pd.DataFrame()
            df = all_tweets[['tweets','model_sentiment','vader_sentiment']]
            data  = [30, 40, 30]
            createPlots(data,1)
            
            all_tweets.to_csv("all_tweets.csv", index= False)

            summary = summary.set_index('name')
            my_summary = my_summary.set_index('name')

            return render_template('index.html',chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis, my_dict= my_dict, my_summary= my_summary.to_html(), summary= summary.to_html())
        else:
            error = "Sorry, no news found. Try other keyword"
            return render_template("welcome.html", error=error)
    

@app.route("/getLocations", methods=['POST'])
def getLocations():
    all_tweets = pd.read_csv("all_tweets.csv")
    print(len(all_tweets))
    #Get Geo Coordinates
    all_tweets = get_coordinates(all_tweets)

    output_file("templates/gmap.html")

    pos_data =all_tweets.dropna()[all_tweets['model_sentiment']=='positive']
    neg_data =all_tweets.dropna()[all_tweets['model_sentiment']=='negative']
    ntl_data =all_tweets.dropna()[all_tweets['model_sentiment']=='neutral']
    map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=3)

    # For GMaps to function, Google requires you obtain and enable an API key:
    #
    #     https://developers.google.com/maps/documentation/javascript/get-api-key
    #
    # Replace the value below with your personal API key:
    p = gmap("AIzaSyBNoWs1e8I7XvEBcA4KbsR8hKP4mJzot2U", map_options)

    source = ColumnDataSource(
        data=pos_data
    )

    p.circle(x="longitude", y="latitude", size=15, fill_color="green", fill_alpha=0.8, source=source)

    source = ColumnDataSource(
        data=neg_data
    )

    p.circle(x="longitude", y="latitude", size=15, fill_color="red", fill_alpha=0.8, source=source)

    source = ColumnDataSource(
        data=ntl_data
    )

    p.circle(x="longitude", y="latitude", size=15, fill_color="blue", fill_alpha=0.8, source=source)
    show(p)

    return render_template("locations.html")

if __name__ == "__main__":

	app.run(debug= True,host='0.0.0.0',port = 5000,threaded=True)