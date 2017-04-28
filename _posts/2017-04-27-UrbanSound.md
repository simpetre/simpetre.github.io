---
layout: post
title: Audio signal processing and convolutional neural networks
icon: volume-up
---

This project is an example of image classification (more specifically, the techniques associated with image classification). The [UrbanSound dataset](http://www.telegraph.co.uk/travel/destinations/europe/united-kingdom/england/london/articles/Which-foreign-passports-are-most-common-in-London/) is available at that link, and is a labelled collection of recordings of ten different typical city sounds (dogs barking, emergency vehicle sirens, children playing...), and attempted classification of these sounds.

## Audio Processing
There are a few steps to this particular classification problem - to start, let's tackle the problem of getting the sound in a format that is useful for computers. We need to convert what is effectively an analog signal into its digital counterpart to feed into some sort of machine learning algorithm for classification. The signal is usually broken up into small (overlapping) chunks (this is called sampling) - and then the chunks are taken from the time domain into the frequency domain by applying a mathematical transformation called a *Fourier transform*. this transformation produces a vector of complex numbers. These numbers can be stacked against each other vertically, to produce an image of the audio signal - with time along the *x*-axis, frequency along the *y*-axis and intensity of sound represented by the colour at that point. This is known as an audio spectrogram.

## Convolutional neural networks
So - we've converted an audio classification problem into an image classification problem. Image classification is a hot topic right now - massive tech companies are devoting a lot of time and resources to improving their image recognition technology (facial recognition and computer vision, as examples).
