---
layout: post
title: Natural language processing using LexRank
icon: book
---

LexRank is an article summarisation technique explored by Erkan and Radev in their [2004 paper](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html) on the subject. It uses the concepts introduced by PageRank, which is the well known algorithm that is the backbone of the Google Search methodology, and extends them to the field of text. For this project, I implemented LexRank in Python, and applied it to the New York Times corpus to see what sort of results I'd get.

## Natural Language Processing

NLP is a field of machine learning related to the processing of written or spoken speech by a computer. Some examples are chat bots - in the form of Facebook advertising bots interacting with their customers; spam filtering on email platforms; classification of news articles - or automatic text summarisation, which is what this project is about.

There are two forms of text summarisation - *abstractive* and *extractive*. Extractive summarisation attempts to distil the text down into its most explanatory or useful phrases (and is what this project uses), while abstractive summarisation summarises the text in new words. Abstractive summarisation is a much more difficult problem and is still very much in its infancy.

## PageRank, and the idea of prestige

PageRank, which was introduced by Larry Page and Sergey Brin in their [1999 paper](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) on the topic, explores the idea of ranking websites based on the importance of the sites that reference them. For a particular website being ranked, the "better" (or higher *prestige*) the sites that feature links to the site being ranked, the better that site will do in the rankings. In (maybe over) simple terms - if an important website features a particular link, then that particular link must itself be important.

## Prestige in text summarisation

This is a concept that Erkan and Radev have applied to the problem of text summarisation. If we look at the problem we are trying to solve as finding the subset of sentences within a body of text that do the best job of summarising the main theme of the text, then we can consider the importance of particular sentences, and their relationship with other sentences within the body of text (or corpus). A sentence, in this context, has a higher importance than another sentence if it is more similar to other sentences in the corpus than the other sentence is.
