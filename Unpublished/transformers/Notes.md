# Transformers


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