
# %%
"""
# Text classification

TLDR:
*   Concat all sentences to single long list with list of starting offsets
*   Convert words to tokens
*   Tokens + offsets -> nn.EmbeddingBag layer
*   which converts to embeddings per token and then aggregates
*   Outputs tensor of shape: [No of sentences, embedding size]

Take arbitrary length sequence of words and classify the output.
E.g. news headline, classify as sport/financial etc.:
x = "Wall St. Bears Claw Back Into the Black (Reuters) Reuters -
Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green
again."
y = 3 (financial)

Each string (sequence of words) is split into tokens.
Each word is given an unique numerical ID (token).
Tokens come from a vocab - massive list of words assigned a unique ID.
Now each string is a list of ints.

For each training example x, y we therefore get:
[[]]

Each training sentence, x, can be different length.
We cant represent with a tensor/matrix without knowing the length of the longest sentence.
And in which case it would become sparse.

We concat all the sentences together into a long list of words/tokens.
We keep a list of the start of each sentence within the long list of words:

x[0] = "sentence no 1"
x[1] = "sentence no 2"
==> ["sentence", "no", "1", "sentence", "no", "2"], [0, 3]
Where 0 and 3 are the indexes for the start of each sentence.
The long list of words is presented in its token form:
["sentence", "no", "1", "sentence", "no", "2"]
==> [7, 43, 67, 7, 43, 68]

To train on the tokens would be inefficient as we have too many unique tokens.
E.g. one hot encoding of the whole vocab would give a feature set too large.
Transform those tokens in a compact representations, embeddings.
Similar to one hot encoding and then applying PCA to reduce the dimensional.

These two lists (concat of sentences, and list of starting offsets) are fed into an nn.EmbeddingBag layer
which converts the sentence tokens into embeddings and then aggregates them via mean/sum etc..
The output of the embedding layer is:
[No of sentences, embedding size]
or
[TrainingExample, features]

This output is appropriate to flow into linear layers etc..


source: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
"""

# %%
import torch

class TextClassificationModel(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
model = TextClassificationModel(10, 5, 3)

from torchinfo import summary

tokens = [7, 43, 67, 7, 43, 68]
offsets = [0, 3]
summary(model, input_data=(torch.tensor(tokens), torch.tensor(offsets)))

# %%
"""
# Word classification with RNNs

Take single words of artitrary length. Classify the output.
E.g. take surnames and classify which language they are likely from.
x = "Smith"
y = "English"

Convert the surname into tokens and one hot encode them.
Each surname becomes:
[no_letters, letter_vocab_size]
no_letters = Smith->5
letter_vocab_size = 26 for lowercase + punctuation

Each row is passed through an RNN layer one by one.
This updates the RNN hidden state.
The final hidden state is passed through a linear layer to predict the output class.

For RNNs it makes sense to me atm to use input data in batch first mode.
I.e.
[batch_idx, no_letters, letter_vocab_size]
We need to reset the hidden state for each sequence of letters.
Therefore the batch size is always 1 here.
This makes training slow.

source: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
# %%

# %%
"""
# General tips

*   Datasets
    *   should output in a readable ways
    *   I.e. iteration outputs a sentence in string form, rather than tokens
DataLoaders
"""