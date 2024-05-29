---

layout: post  
title: Diffusion Models - A PyTorch Implementation  
icon: paint-brush  

---

AI was supposed to come for the creatives last. Our time was supposed to be freed up from mundane tasks like coding and calculating so that we could paint, draw, write, sculpt, and dance. Sadly, some clever people have figured out how to build *generative* models that take a text prompt and return a high-resolution image from it in a matter of seconds, leaving swathes of people who do creative tasks for a living at the least suffering from an existential crisis, and at the most out of jobs. This has sparked a secondary, and very interesting discussion about ownership and ethics, but that's a topic for a different post. For now, let's delve into the nuts and bolts.

## Overview

The state-of-the-art (SOTA) for image generation models is the *diffusion* model. This model framework can be broken down into several steps:

1. Text encoding
2. Forward diffusion
3. Reverse diffusion

Diffusion models begin by taking a text input (e.g., "an image of a beautiful sunset") and using a pre-trained model (often a transformer-based model like CLIP) to project this down into a latent space. The model then takes the latent vector representation of the text and applies progressive steps of Gaussian noise to it, learning how to remove this noise to recover the original image. At inference time, the model starts with a random latent noise vector, conditions it on the latent text representation, and gradually "removes" the noise to recover a beautiful representation of the input text. While this process might sound like black magic, it is grounded in robust mathematical principles and involves several hyperparameters that require careful tuning for optimal results.

Sounds intriguing, right? Let's investigate further.

## Text Encoding

Diffusion models have their basis in physics, in the ideas of thermodynamic diffusion. Imagine we drop a single droplet of ink into a bucket of water. Instantaneously as the ink hits the water's surface, the behavior of the ink is fairly predictable. If we move forward in time gradually, the ink dissipates further through the bucket, until it's homogeneously mixed, and we don't stand a chance of being able to recover the dynamics of the ink or the bucket immediately after the droplet hits. If we slice time up into discrete pieces though, we can learn to predict the previous step's dynamics based on the current step. This is what we're doing when we use a diffusion model - we take seeming random noise and progressively remove this noise step-by-step until hopefully a coherent image is recovered. For our use case, we need some way to be able to steer the noise removal process in the right direction, though - otherwise, how will our model know whether underneath the noise, it's looking for a sunset, children playing hopscotch, or a knight at a disco?

As mentioned, we usually use a pre-trained transformer at this step, like OpenAI's CLIP (Contrastive Language-Image Pre-training). CLIP's contrastive training is such that it takes in labeled images and projects both the text labels and the images down into a lower-dimensional latent space, such that images occupy a similar position to their corresponding labels, semantically similar text occupies similar positions to each other (e.g., "a sunset over a mountain range" or "an image of a beautiful sun setting over mountains"), and semantically different text (or text labels for different images) occupy different positions in the latent space.

It achieves this by using two transformers, one on the text input and one on the image input, to project the inputs down into a latent space with the same dimensionality, then minimizes the distance between embeddings of corresponding pairs, and maximizes the distance between non-matching pairs using InfoNCE loss.

### Text Transformer

The text transformer in CLIP is responsible for processing the text input. Here are the key points:

1. **Tokenization and Embedding**:
   - The input text is tokenized and each token is embedded into a high-dimensional vector space.

2. **Positional Embeddings**:
   - Positional embeddings are added to the token embeddings to encode the order of the tokens. This ensures that the transformer can capture the sequence information of the text.

3. **Transformer Layers**:
   - The text embeddings (with positional information) pass through multiple transformer layers. These layers use multi-head self-attention and feed-forward networks to learn contextual relationships between the tokens.

4. **Latent Space Representation**:
   - Instead of projecting back into the original text space, the output is kept in the latent space. This latent representation is aligned with the image representation during training.

### Vision Transformer (ViT)

The Vision Transformer processes image inputs and has some unique aspects compared to the text transformer:

1. **Patch Extraction**:
   - Given an image of dimension $H \times W \times C$ (height, width, channels), the image is divided into $P \times P$ patches.

2. **Patch Embedding**:
   - Each patch is flattened and embedded into a high-dimensional vector, resulting in vectors of dimension $P^2 \cdot C$.

3. **Positional Embeddings**:
   - Learnable positional embeddings are added to these patch embeddings to encode their spatial positions within the image.

4. **Class Token**:
   - A special learnable class token is added to the sequence of patch embeddings. This token will aggregate information from all patches through the attention mechanism.

5. **Transformer Layers**:
   - The sequence of patch embeddings (including the class token) is processed through multiple transformer layers. These layers use multi-head self-attention to learn relationships between different patches and the class token.

6. **Aggregating Information**:
   - The class token aggregates information from all patches and captures high-level semantic information about the image. The final representation of the class token is used for alignment with the text embedding.

These elements work together to enable the model to capture detailed visual information and high-level semantics, ensuring a comprehensive understanding of the input data.

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for class token

    def forward(self, x):
        return x + self.pos_embedding

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.drop1 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop1(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEmbedding(self.patch_embed.num_patches, embed_dim)
        self.pos_drop = nn.Dropout(0.1)

        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
        for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(self.pos_embed(x))

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

# Example usage:
model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=1000)
dummy_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
output = model(dummy_input)
print(output.shape)  # Should output: torch.Size([1, 1000])
```

We use the text transformer and the vision transformer to project the inputs down into a latent space with the same dimensionality, and then we train the model using InfoNCE loss to minimise the distance between text and label, and maximise the other distances in the model. 

### InfoNCE Loss

CLIP uses a contrastive loss known as InfoNCE (Information Noise-Contrastive Estimation) to align the representations of images and their corresponding text descriptions in a shared latent space. The main idea is to maximize the similarity between matched image-text pairs while minimizing the similarity between unmatched pairs.

### Steps to Compute InfoNCE Loss for CLIP

1. **Compute Embeddings**:
   - Let $ \mathbf{v}_i $ be the embedding of the $i$-th image.
   - Let $ \mathbf{t}_i $ be the embedding of the $i$-th text.

2. **Normalize Embeddings**:
   - Normalize the embeddings to have unit length. This is done to ensure that the dot product is equivalent to cosine similarity.
   
   $$
   \mathbf{v}_i \leftarrow \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}
   \mathbf{t}_i \leftarrow \frac{\mathbf{t}_i}{\|\mathbf{t}_i\|}
   $$

   This step ensures that the dot product of the vectors will represent the cosine similarity. It helps in stabilizing the training process and makes the similarity computation invariant to the scale of the embeddings.

3. **Compute Similarity Scores**:
   - Compute the pairwise cosine similarity between all image and text embeddings in the batch.
   $$
   S_{ij} = \mathbf{v}_i \cdot \mathbf{t}_j
   $$
   where $ S_{ij} $ is the similarity score between the $i$-th image and the $j$-th text.
The similarity score $ S_{ij} $ is the dot product between the normalized embeddings of the $i$-th image and the $j$-th text. High scores indicate high similarity.

4. **Compute Logits**:
   - Scale the similarity scores by a temperature parameter $ \tau $ (learnable or fixed).
   $$
   \mathrm{logits}_{ij} = \frac{S_{ij}}{\tau}
   $$

5. **Compute Cross-Entropy Loss**:
   - For the images:

     $$
     L_{\text{img}} = \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathrm{logits}_{ii})}{\sum_{j=1}^N \exp(\mathrm{logits}_{ij})}
     $$

   - For the texts:

     $$
     L_{\text{text}} = \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathrm{logits}_{ii})}{\sum_{j=1}^N \exp(\mathrm{logits}_{ji})}
     $$

   - Combine the losses:

     $$
     L = \frac{1}{2} (L_{\text{img}} + L_{\text{text}})
     $$

        - For each image $ \mathbf{v}_i $, the model should assign high similarity to the corresponding text $ \mathbf{t}_i $ and low similarity to all other texts $ \mathbf{t}_j $ ($ j \neq i $).
   - The same applies for each text $ \mathbf{t}_i $, where it should assign high similarity to the corresponding image $ \mathbf{v}_i $ and low similarity to all other images $ \mathbf{v}_j $ ($ j \neq i $).
   - The cross-entropy loss computes the discrepancy between the predicted similarity scores and the ideal distribution where only the correct image-text pairs have high similarity.

### Implementation in PyTorch

Here is an example of how you might implement this loss in PyTorch:

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeddings, text_embeddings, temperature=0.1):
    # Normalize the embeddings
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity scores
    logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    logits_per_text = logits_per_image.t()
    
    # Generate ground truth labels
    batch_size = image_embeddings.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
    
    # Compute cross-entropy loss
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_text = F.cross_entropy(logits_per_text, ground_truth)
    
    # Return average of the two losses
    return (loss_img + loss_text) / 2

# Example usage
image_embeddings = torch.randn(32, 512)  # Example image embeddings
text_embeddings = torch.randn(32, 512)  # Example text embeddings
loss = clip_loss(image_embeddings, text_embeddings)
print(f'CLIP Loss: {loss.item()}')
```

## Forward Diffusion

Forward diffusion is the ink dissipating in the water over successive time slices. We take an image, and successively add Gaussian noise to it, sampled from the distribution 

$$
   q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$
   where:
   - $x_t$ is the noisy data at time step $t$.
   - $\beta_t$ is a variance schedule, a small positive constant that determines the amount of noise added at each step.
   - $\mathcal{N}$ denotes a Gaussian distribution.
   - $I$ is the identity matrix.
