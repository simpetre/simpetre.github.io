---
layout: post
title: Transformers - A PyTorch Implementation
icon: book
---

It seems like everywhere you look these days, you see AI - people talking about AI, or implementing AI, or trying to cash in on the AI goldrush. There are a few technologies underpinning this recent trend, but by far the most influential one is the concept of a *transformer*, which is the building block that you'll find basically every single Large Language Model (unless you've been living under a rock you what a LLM is - think ChatGPT, Claude, Meta's LLaMa 3). Let's talk about transformers in greater detail, and implement a transformer architecture using PyTorch.

## Historical Context

Prior to transformers, NLP was dominated by recurrent neural networks (RNNs) and techniques based on RNNs. These techniques faced several issues that limited their impact:

- **Vanishing/Exploding Gradients**: RNNs are sequential models, and during backpropagation, the gradients are passed back through several time-dependent layers of the neural network. If the magnitudes of these gradients are either too large or too small, they can "explode" (tend towards infinity) or "vanish" (tend towards zero) due to the magnification effect of repeated multiplication. Techniques can mitigate this (e.g., initialization, specific activation functions), but they're not trivial to implement.

- **Sequential Processing**: RNNs are sequential models. This means that the next token is determined based on the previously calculated tokens; weights for token $i$ must be calculated before those for token $i+1$, $i+2$, and so on. This means modern hardware (like GPUs') parallelisation capabilities can't be leveraged, which means training times are long and expensive.

- **Long Contexts**: Managing long contexts is difficult. If a sentence contains a token at its beginning and another reference at the end after a long passage of text, models might struggle to recall the beginning of the sentence. Techniques like LSTMs have been used to manage this, but they often require careful tuning.

Transformers avoid many of these issues. They are parallelizable and designed to avoid vanishing/exploding gradients.

# Understanding Transformers and Self-Attention

Transformers are mainly used in natural language processing (NLP). The aim is to take language, transform it into numbers, extract the semantic meaning from these numbers, and then generalize this meaning to unseen data. We begin with a corpus of text that we want to use to train a model. Here's the process (at least, in terms of causal language models):

1. Tokenize the text
2. Pass the tokenized text through an embedding layer
3. Pass the embedded text through a self-attention layer
4. Pass the output of the self-attention layer through a feed-forward neural network (FFNN)
5. Apply batch normalization and residual connections

## Tokenization

### Byte-Pair Encoding (BPE)

```
import collections
import re

def get_vocab(corpus):
    vocab = collections.Counter()
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] += 1
    return vocab

corpus = [
    "this is a sample sentence",
    "tokenisation is important",
    "we are learning about bpe"
]

vocab = get_vocab(corpus)
print(vocab)
```

We start by breaking down the vocabulary to the character level (e.g., `t h i s`, `i s` ...). 

```
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    pattern = ' '.join(pair)  # Create the pattern to replace
    replacement = ''.join(pair)  # Create the replacement string
    for word in v_in:
        w_out = word.replace(pattern, replacement)  # Replace the pattern with the replacement
        v_out[w_out] = v_in[word]
    return v_out

num_merges = 50  # Number of merges to perform
bpe_merges = []
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    bpe_merges.append(best)
    print(f'Merge {i + 1}: {best}')
    print(vocab)
```

From here, we look at the frequencies with which pairs of tokens occur and group together the ones with the highest incidence. For example, in the above toy corpus "i s" occurs most frequently, so we merge these separate tokens together into a new `is` token and the corpus is now `t h is`, `i s`, ... `t o k e n is a t i o n` and so on. We continue this process iteratively, combining the most frequent token pairs until our corpus reaches the desired size.

```
def encode(token, bpe_merges):
    token = ' '.join(list(token)) + ' </w>'
    chars = token.split()

    i = 0
    while i < len(chars) - 1:
        pair = (chars[i], chars[i + 1])
        if pair in bpe_merges:
            chars[i:i + 2] = [''.join(pair)]
        else:
            i += 1

    return ' '.join(chars)

# Example of encoding
encoded = encode("miss", bpe_merges)
print("\nEncoded 'miss':")
print(encoded)
```

Once we've used BPE to create our collection of tokens, we can use our work to tokenise new text.

## Embedding

The initial representation of text is sparse. We have a corpus of documents with a vast vocabulary (English has around 170,000 words in use), and we indicate to a ML model which words are present in a document through one-hot encoding - we represent an input *context* using a N-dimensional vector of ones or zeroes, where a 1 at position *i* indicates that word *i* is present in our text. This isn't super useful for a computer - it doesn't give us anything about how words are related to each other, or what the *meaning* underpinning words is. We can compress these word representations into a dense "embedding space" where semantically similar words occupy similar positions - so we get closer to our toy example of `king - man + woman = queen`.

```
import torch

# Parameters
vocab_size = 10000  # number of unique tokens
embedding_dim = 300  # size of each embedding vector

# Step 1: Initialize the embedding matrix with random weights
embedding_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True)  # Enable gradient

# Step 2: Function to retrieve embeddings for given indices
def get_embeddings(indices):
    return embedding_matrix[indices]

# Example usage
input_ids = torch.tensor([1, 2, 798, 1253, 9999], dtype=torch.long)  # Indices of tokens
embeddings = get_embeddings(input_ids)

print(embeddings)  # Output the embedding vectors for the input indices

```

The embedding step creates a matrix with dimensionality $N \times D$, where $N$ is the dimension of the original, sparse vocabulary, and $D$ is the dimension of the dense embedding space. Backpropagation produces a matrix of weights, such that row $n$ in the lookup table is the $D$-dimensional representation of token $n$ in the embedding space.

## Position-Wise Encoding

By design, transformers don't consider the order of tokens by default. To combat this, Vaswani et al. used position-wise encoding to reveal token position to the network. An $N \times D$ matrix that encodes position is created and added to the embedding layer before the first self-attention layer.

$$
PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/D}}\right)
$$

$$
PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/D}}\right)
$$

These formulas assign a unique value to each element of the positional matrix, which is added to the encoding matrix. This allows the transformer to learn to consider position when it's training or performing inference.

## Self-Attention

This is the fun bit! Self-attention has been driving the AI revolution of the last few years and is the interesting part of large language models (LLMs) like ChatGPT. The process is as follows:

1. We take three $D \times D'$ dimensional matrices, $M_Q$, $M_K$, and $M_V$. Often $D' = D$, so we'll use that simplification here.
2. Recall that our training text is $N \times D$, where $N$ is the number of tokens in the sample data and $D$ is the dimensionality of the vector space. We multiply each of the three matrices $M_Q$, $M_K$, and $M_V$ together, to get the three matrices $Q$, $K$, and $V$, for Query, Key, and Value, respectively. Each of these matrices retains the dimensionality $N \times D$.

Each of these matrices can be considered in the following context for token $n$:
- **Q (Query)**: What information am I looking for from other tokens?
- **K (Key)**: What information do I have to give to other tokens?
- **V (Value)**: How much do I want my information to contribute to the final output?

The output of the attention step is as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) \cdot V
$$

We take the dot product of each token's query and key vectors to get an $N \times N$ matrix. If the $Q$ vector of token $x_1$ and the $K$ vector of token $x_2$ are similar, the value at $(x_1, x_2)$ will be close to 1. We normalize these values using the square root of the embedding space dimensionality, pass them through a softmax function so they sum to 1, and use these values to scale the contribution of each value to the final output.

## Feed-Forward Neural Network (FFNN)

Values from the attention step are passed into a FFNN. Usually, this neural network has three layers: a linear layer, a ReLU activation, and another linear layer. The FFNN is layer-normalized, and a residual connection bypasses the network to help with stability.

1. **First Linear Layer**:
   $$
   \text{Linear}_1(x) = xW_1 + b_1
   $$
   Followed by a ReLU activation.

2. **Second Linear Layer**:
   $$
   \text{Linear}_2(x) = xW_2 + b_2
   $$

## Residual Connections and Layer Normalization

Additionally, the transformer architecture includes residual connections and layer normalization after each sub-layer (attention and FFNN). These steps help stabilize training and improve convergence:

$$
\text{LayerNorm}(x) = \frac{x - \text{mean}(x)}{\text{std}(x)}
$$

## Putting It All Together

By stacking multiple layers of attention mechanisms and FFNNs, the transformer model can build rich representations of the input data, capturing long-range dependencies and complex interactions between tokens.

---

This document provides a comprehensive overview of how transformers and self-attention work, covering their historical context, tokenization, embedding, attention mechanisms, and feed-forward neural networks.
