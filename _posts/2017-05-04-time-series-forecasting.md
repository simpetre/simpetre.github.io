---
layout: post
title: Time series forecasting and tweet frequencies
icon: line-chart
---

This post is an example of time series forecasting, using a time series created from Twitter search term frequencies. Twitter have recently opened up their searches, so that tweets back to the beginning of the service are returned as results for a search. Unfortunately the Twitter API, while useful to an extent, doesn't benefit from this change. I used a Python library called [twitterscraper](https://github.com/taspinar/twitterscraper) that uses BeautifulSoup to scrape the search results from the site.

### Creating the time series
Implementation is quite simple. I wanted to get tweet frequency, and Twitter timestamps tweets to the second - so I took the tweets for a particular search term (let's look at Apple, here), divided them into buckets and stored them in a Counter object.

```python
import twitterscraper as ts

def count_and_bucket_tweets(query):
    tweet_ts = Counter()
    for tweet in ts.query_tweets(query):
        tweet_ts[str(tweet.timestamp)[:-3]] += 1
```
