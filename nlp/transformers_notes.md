# Transformers

## Tokenisation
Tokenisation is where we encode sentences of text into lists of integers, ready for embedding.

GPT-2 as others use Byte-Pair encoding. It encodes each letter (and unicode) as a based vocabulary.
Then common pairs of letters are merged and added to the vocabulary.
This is repeated with triple of letters until the vocabulary reaches an ideal size.
It can encode any artitary string without loss.

GPT-2 encoding is given here: https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
and has 50k tokens.

Ref:
* https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
* https://jaykmody.com/blog/gpt-from-scratch/

## Embedding layer
Tokens are converted to a vector representing an embedding.
The embeddings are learnt during the transformer training.

Transformers are invarient to the input token ordering. Shuffling will give the same output.
Therefore we add positional encodings to the word embeddings to address ordering.
Typically we use cos/sin pairs for a range of frequencies (rather than trained parameters) to uniquely encode each position in a sequence.

## Attention
Attention uses values from all input tokens/embeddings in the sequence to compute the output for a given input token.
Each embedding vector in a sequence goes through a linear layer ($w_v$) to obtain a *Value*.
The output from any given embedding vector in a sequence is a weighted sum of all the *Values* in the sequence.
The weights come from a softmax from a *Query* and *Key*.
The *Query* is taken from the embedding vector through a linear layer ($w_q$).
A *Key* is computed for each vector in the sequence using another linear layer ($w_k$).
The weights are the dot product of the *Query* with each *Key* and then normalised with a softmax:

$$
\mathrm{attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,
$$
where $d_k$ is the lenght of the sequence.

As such we have 3 sets of weights for linear layers - *Value*, *Query* and *Key*.

Ref:
* https://jaykmody.com/blog/attention-intuition/#word-vectors-and-similarity
* https://jaykmody.com/blog/gpt-from-scratch/
* https://lilianweng.github.io/posts/2018-06-24-attention/
* https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

## GPT
In addition to attention:
* Decoder only - there is no encoder used. We auto-regressing to append outputs to the inputs to generate the next output token.
We can sample from the output layer, which gives logits for each token, to give stochasic outputs. Using the temperature of the softmax will change how deterministic the output sequence is.

* Self-attention - attention is used within the sequence to model based on the vectors within the sequence. The *Query*, *Key* and *Value* are all calculated from the same input sequence.

* We have a causal mask applied to ensure the attention weights only look at vectors in the sequence prior to the current vector.
This avoids learning from the subsequent vectors in the sequence.

* Residual connections - attention layers are stacked. To reduce the problem of vanishing/exploding gradients, the attentions outputs are added to the original embedding vectors each time to allow a residual connection.

* Multi-head attention - we use multiple attention blocks in parallel, with different $w_q$, $w_k$, $w_v$ weights for each. The heads are concatenated at the end and a linear layer projects back down the embedding space dimension.

## Article 1
`https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/`

### Word embeddings
As in other applications of NLP. Embedding layer which takes in word vectors. Word vectors come from sentences converted into numerical tokens.
```
output_length = 6
word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
print(embedded_words)
```

### Position encodings

Positional encoding for spatial distance, near by words higher values.

Create embedding layers
Instead of feeding in word vectors, send in range of indicies
I.e.

```
position_embedding_layer = Embedding(output_sequence_length, output_length)
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
print(embedded_indices)
```

Typically we use cos/sin pairs for a range of frequencies.

### Output
Add the word and position embeddings together:
```
final_output_embedding = embedded_words + embedded_indices
```

## Other
https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention


Training batch:
64 x 38 = 64 examples of max length 38 words
[[word 1 word 2 word 3. ...]
 [word 1 word 2 word 3. ...]]

Convert each example from the training batch into input data X
X = ['one word at a time']
tokenizing converts each word to a int, incl spaces and padding to a set input sequence length.
X = [1, 2, 3, 4, 5, 6, 7, 8]
embedding makes each token into a vector. Allows similar words to be similar.
X = [[1, 4, 5, 6, 7], [1,3, 5, 7, 8], [1, 3, 6, 7, 8] ... ]
X = [[1 2  4 5],   [word 1 embedding]
    [1 2 4 5 5]     [word 2 embedding]
    [11 2 3 4]]     [word 3 embedding]

Positional encoding for spatial distance, near by words higher values
X = f(X)?
Masking - binary mask for padding, applies to training examples where they were shorter than the max length, so that whole word is masked
mask = []?
Look ahead mask - for each vector limits what inputs are used
Both masks multiplied by the input matrix X


Value matrix - standard linear layer
V = xW
applies to single input word (embedded vectors) at a time. Weighted sum of word embedding.
Rows of V = [[emb_word1*W1, emb_word1*W2]
             [emb_word2*W1, emb_word2*W2]]

temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

masks applied within the attention weights
attention weights = [0, 0.2, 0.5, 0.3] - one per word
attenweights * V = output of layer
output = [[attentionQ1*V[:,1], attentionQ1*V[:,2]]
          [attentionQ2*V[:,1], attentionQ2*V[:,2]]]

Multihead attention - duplicating the attention with different V, K, Q
Splits the value weights depth into multiple attention heads

Training
learning rate schedule like previous mentioned - low at start to estimate adam variances?
Teacher forcing - autoregressive model - regardless of the output, we feed in the correct output for the next step
Look ahead masking to prevent the model learning the output directly


# Outstanding questions:
* Why multihead outperforms single attention
* Feeding outputs to training cycles - not fully understood