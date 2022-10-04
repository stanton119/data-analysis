# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

# TD IDF - count the occurences of each word and scale by the frequency of that word in the whole document/corpus

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)

X.todense()

print(X.shape)