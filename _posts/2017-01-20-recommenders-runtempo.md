---
layout: post
title: RunTempo - getting fitter using recommender systems
icon: music
---

Recently I attended a data science program in San Francisco run by a company called Galvanize. For the last two weeks of of the program we worked individually on a capstone project that showcased the skills we'd learned to that point. My project is called RunTempo - it produces a playlist of songs for a user to exercise to, where the tempo of the songs is matched to the speed that the user wants to exercise at.

The data I used for my project was sourced from the LabRosa Million Songs Dataset, and I used a matrix factorisation collaborative filtering recommender, built using Turi GraphLab Create.

## Motivation
Recommender systems are so hot right now - ubiquitous in data science, at the core of many top-tier tech offerings, like Amazon's product recommendations, the Netflix algorithm, Spotify's discover playlists - the list goes on. At the same time, I like to exercise, and I like to listen to music. I thought bringing these two together and using a recommender system to build something around them would be a good opportunity to showcase something with a nice amount of data science and machine learning meat, while at the same time helping keeping me motivated by something that I'm interested in doing.

## Data
I pulled data from several sources when creating RunTempo. The base recommender was built with the [Taste Profile](http://labrosa.ee.columbia.edu/millionsong/tasteprofile), which is a sister dataset to LabRosa's [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) (MSD). The Taste Profile contains user/song/play count triplets for 1019318 unique users and 384546 unique songs for a total of 48373586 samples. As a first pass, I used play count as a proxy for rating - I assumed that the more a particular user listened to a particular song, the more the user liked it, and the higher they would rate it.

## Model
The model I settled on was a matrix factorisation recommender optimised for ranking, created using [GraphLab Create](https://turi.com). I fed in the data unaltered initially and got some pretty questionable results, so I amended the target variable to map each user's playcount to a rating out of five (I took individual user's playcounts, determined the quintiles those playcounts fell into for that particular user and then used this quintile as a rating out of five) and this is what I based my final model on.

To predict a playlist for a particular user I chose three random songs from the entire dataset, and the three respective songs that were as dissimilar to those songs as possible (i.e. were furthest away in terms of Jaccard distance) and asked the user to choose their preference out of these three pairs. I used this preference as a proxy for rating (a selected song was assumed to be rated five out of five by the user) and then I found the user that was closest to a user that had these three ratings (again in terms of Jaccard distance) to determine the ranking order of the songs in the dataset. I filtered these songs on the appropriate tempo range and return them to the user in increasing order of tempo (within a 20 footsteps or beats/minute range).

## Webapp
I chose to present RunTempo in the form of a website. The user chooses from three different options for speed and selects their choice of songs, presented as three sets of two options, and the RunTempo creates and renders the playlist dynamically. If you want to check it out, fork it from my [Github repo](https://github.com/simpetre/runtempo).

## Future Work
The main obvious deficiency of RunTempo is that the music that it's based off is (in musical years) very old - the MSD was created in early 2011, and songs that it contains were released in 2010 or earlier. The next iteration of the project I'll be working on is to take some actual sound data for the songs in the base recommender (Spotify publish a number of derived metrics, such as "speechiness", "acousticness", "loudness", "valence" - which all have pretty self-explanatory meanings, except for valence, which is a measure of how "happy" the song sounds), and use a clustering algorithm to find popular songs (the most recent ten thousand songs that have been featured in the Billboard 100 charts) that are close to these.
