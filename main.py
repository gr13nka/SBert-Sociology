import tweepy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np


#Tweeter stuff
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Func for fetching tweets 
def fetch_tweets(query, max_tweets):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(max_tweets):
        tweets.append(tweet.full_text)
    return tweets

query = "your_search_term"
max_tweets = 100
fetched_tweets = fetch_tweets(query, max_tweets)


# Load sbert
model = SentenceTransformer('all-MiniLM-L6-v2')
# generate embeddings 
embeddings = model.encode(tweets)

# We have to determine num of clusters, remember the elbow method son?
num_clusters = 5  # Set based on elbow method
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

# Analyze clusters

clustered_tweets = [[] for i in range(num_clusters)]
for tweet_id, cluster_id in enumerate(cluster_assignment):
    clustered_tweets[cluster_id].append(tweets[tweet_id])

# Review 
for i, cluster in enumerate(clustered_tweets):
    print(f"Cluster {i+1}")
    print(cluster[:5])  # Displaying first 5 tweets in each cluster
    print("\n")




