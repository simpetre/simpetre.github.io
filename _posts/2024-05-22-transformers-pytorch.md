---
layout: post
title: Transformers - A PyTorch Implementation
icon: book
---

It seems like everywhere you look these days, you see AI - people talking about AI, or implementing AI, or trying to cash in on the AI goldrush. There are a few technologies underpinning this recent trend, but by far the most influential one is the concept of a *transformer*, which is the building block that you'll find basically every single Large Language Model (unless you've been living under a rock you what a LLM is - think ChatGPT, Claude, Meta's LLaMa 3). Let's talk about transformers in greater detail, and implement a transformer architecture using PyTorch.

## Historical Context

Prior to transformers, NLP was dominated by recurrent neural networks (RNNs) and techniques based on RNNs. These techniques faced several issues that limited their impact:

- **Vanishing/Exploding Gradients**: RNNs are sequential models, and during backpropagation, the gradients are passed back through several time-dependent layers of the neural network. If the magnitudes of these gradients are either too large or too small, they can "explode" (become excessively large, leading to instability) or "vanish" (tend towards zero).

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

Byte-Pair Encoding (BPE) is a popular tokenization method that balances the granularity between character-level and word-level tokenization, handling rare and common words efficiently.

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

The embedding step creates a matrix with dimensionality $N \times D$, where $N$ is the dimension of the original, sparse vocabulary, and $D$ is the dimension of the dense embedding space (here $N = 10000$ and $D = 300$). Backpropagation produces a matrix of weights, such that row $n$ in the lookup table is the $D$-dimensional representation of token $n$ in the embedding space. During training and backpropagation, the model will learn the optimal embeddings.

## Position-Wise Encoding

Because transformers process tokens in parallel, they need a way of determining the order in a sequence. We can use position-wise encoding (in the same way that Vaswani et al. did in their seminal paper).

An $N \times D$ matrix (where $N = 10000$ and $D = 300$) that encodes position is created and added to the embedding layer before the first self-attention layer. Each element $i, j$ in the matrix will be unique, allowing the model to learn these values during training and backpropagation and to distinguish between different positions.

The position-wise matrix and the embedding matrix have the same dimensions, but this is primarily for convenience. The position $i, j$ in one matrix doesn't impart any information about the position $i, j$ in the other; in fact, the information in each matrix is orthogonal. By introducing this form of "structured noise" into our model, we enable it to learn positional information during training.

The positional encoddings are given by the following equations:

$$
PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/D}}\right)
$$

$$
PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/D}}\right)
$$

This is implemented in the below code:

```
import torch
import math

def positional_encoding(max_seq_length, embedding_dim):
    """ Generate and return positional encoding.
    Args:
        max_seq_length (int): Maximum length of the input sequences.
        embedding_dim (int): The dimensionality of the output embeddings (and positional encodings).

    Returns:
        torch.Tensor: The positional encodings (max_seq_length, embedding_dim).
    """
    # Initialize a matrix of shape [max_seq_length, embedding_dim]
    position_encoding = torch.zeros(max_seq_length, embedding_dim)
    
    # Compute positional encodings
    for pos in range(max_seq_length):
        for i in range(0, embedding_dim, 2):
            position_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
            if i + 1 < embedding_dim:
                position_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/embedding_dim)))
    
    return position_encoding

# Example usage:
max_seq_length = 100  # maximum length of sequences
embedding_dim = 300  # dimensionality of embeddings

pos_encoding = positional_encoding(max_seq_length, embedding_dim)
print(pos_encoding.size())  # Output the shape of the positional encoding matrix
```

These formulas assign a unique value to each element of the positional matrix, which is added to the encoding matrix. This allows the transformer to learn to consider position when it's training or performing inference.

## Self-Attention

This is the fun bit! Self-attention has been driving the AI revolution of the last few years and is the interesting part of large language models (LLMs) like ChatGPT. This is the core of the paper by Vaswani et al., the meat and potatoes of most modern LLMs, and has been driving the AI revolution of recent times.

The process is as follows:

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

We calculate the distance between each token's Q and K vectors via dot product, and we store the result in an $N \times N$ matrix. If the $Q$ vector of token $x_1$ and the $K$ vector of token $x_2$ are similar, the value at $(x_1, x_2)$ will be close to 1. We normalize these values using the square root of the embedding space dimensionality (to ensure that the variance of the dot products of the vectors in $Q$ and $K$ stay equal to 1), pass them through a softmax function so they sum to 1, and use these values to scale the contribution of each value to the final output; then we project the dimensionality of the output back to that of the input by passing through a linear layer.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()

        # Define linear transformations for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output linear transformation
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Apply linear transformations
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Compute scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Apply final linear transformation
        output = self.out_linear(attn_output)

        return output

# Example usage
embed_dim = 64
seq_len = 10
batch_size = 32

x = torch.randn(batch_size, seq_len, embed_dim)
self_attention = SelfAttention(embed_dim)
output = self_attention(x)
print(output.shape)  # Should output: torch.Size([32, 10, 64])

```

## Feed-Forward Neural Network (FFNN)

Values from the attention step are passed into a FFNN. Usually, this neural network has three layers: a linear layer, a ReLU activation, and another linear layer. The linear layer projects the output into a dimensionality that's usually some factor higher than the input (e.g. if the input dimensionality is $d$ the output is $4d$). This allows the network to be able to capture richer and more nuanced variations in the input data. Non-linearity is applied in the ReLU step, and then the dimensionality is projected back down into $d$ for the output layer. The FFNN is layer-normalized, and a residual connection bypasses the network to help with stability.

```
import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Save the original input for the residual connection
        residual = x
        # First linear layer followed by ReLU activation
        out = self.linear1(x)
        out = self.relu(out)
        # Second linear layer
        out = self.linear2(out)
        # Add the residual connection and apply layer normalization
        out = self.layer_norm(out + residual)
        return out

# Example usage:
d_model = 512  # Input and output dimensionality
d_ff = 4 * d_model  # Intermediate dimensionality, e.g., 4 times the input size

# Create an instance of the FeedForwardNN
ffnn = FeedForwardNN(d_model, d_ff)

# Create some dummy input data
x = torch.randn(10, d_model)  # Batch of 10 samples, each of dimension d_model

# Pass the input through the network
output = ffnn(x)

print(output.shape)  # Should print: torch.Size([10, 512])

```

## Residual Connections and Layer Normalization

Residual connections, also known as skip connections, are used to improve the flow of gradients during backpropagation. This helps to mitigate the vanishing gradient problem, which is especially important in very deep networks. Layer normalization, on the other hand, helps to stabilize and accelerate the training process by normalizing the inputs across the features.

**Residual Connections**

Residual connections allow the input to bypass one or more layers and be added to the output of those layers. This helps in training deep networks by preventing the gradient from becoming too small (vanishing gradient) as it is backpropagated through many layers.

Mathematically, a residual connection can be described as:

$$
Output = Layer (x) + x
$$

Where $x$ is the input to the layer, and Layer(x) is the transformation applied by the layer.

**Layer Normalization**

Layer normalization normalizes the inputs across the features for each data sample, ensuring that the mean is 0 and the standard deviation is 1. This is particularly useful for NLP tasks where the input sequences can have varying lengths.

The layer normalization operation can be described by the following formula:

LayerNorm(x) = x − mean(x)std(x)+ϵ⋅γ+β

Where γ and β are learnable parameters that allow the normalized output to be scaled and shifted.

```
import torch
import torch.nn as nn

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, d_ff):
        super(ResidualLayerNorm, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Save the original input for the residual connection
        residual = x
        # First linear layer followed by ReLU activation
        out = self.linear1(x)
        out = self.relu(out)
        # Second linear layer
        out = self.linear2(out)
        # Add the residual connection and apply layer normalization
        out = self.layer_norm(out + residual)
        return out

# Example usage:
d_model = 512  # Input and output dimensionality
d_ff = 4 * d_model  # Intermediate dimensionality, e.g., 4 times the input size

# Create an instance of the ResidualLayerNorm
residual_layer_norm = ResidualLayerNorm(d_model, d_ff)

# Create some dummy input data
x = torch.randn(10, d_model)  # Batch of 10 samples, each of dimension d_model

# Pass the input through the network
output = residual_layer_norm(x)

print(output.shape)  # Should print: torch.Size([10, 512])
```

## Putting It All Together

Up to this point we've described how we can use a single attention layer to learn the complex interactions between different tokens in input data, which is often language. Because language is complex and nuanced, we find that if we learn several different representations of the attention weights, we can capture more complex dynamics with our model and get better outcomes. This is called *multi-head* attention (as opposed to the *single-head* attention mechanism that we've been describing to this point). We're effectively ensembling several different representations of the attention weights together - randomness during parameter initialisation in the $Q, K$ and $V$ matrices leads to different values after training and backpropagation. We generate $k$ such different sets of attention parameters (we say $k$ different heads), and concatenate these values together before we project them down into our input dimensionality using our linear layer. 

By stacking multiple layers of attention mechanisms and FFNNs, the transformer model can build rich representations of the input data, capturing long-range dependencies and complex interactions between tokens.

## Conclusion

Transformers and their self-attention mechanisms have revolutionized NLP and extended their reach into various domains, including computer vision and audio processing. Self-attention, as described, allows the model to focus on different parts of the input sequence when generating an output. This contrasts with cross-attention, which is used in tasks requiring the interaction between different sequences, such as in encoder-decoder architectures for machine translation. The ability of transformers to handle long-range dependencies, process sequences in parallel, and leverage multi-head attention makes them versatile and powerful for a wide range of applications. For further reading and a deeper dive into transformers, I recommend the seminal paper "Attention is All You Need" by Vaswani et al. and resources on implementing transformers in PyTorch.
