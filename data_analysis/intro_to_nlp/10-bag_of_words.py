#!/usr/bin/env python3
"""Turn token lists into a Bag-of-Words count matrix."""
import sklearn


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """Build a Bag-of-Words matrix with CountVectorizer.
    Returns (X, vectorizer): the sparse matrix and the fitted vectorizer."""
    # each token list becomes one string
    docs = []
    for tokens in corpus_tokens:
        docs.append(" ".join(tokens))

    # the tokens are ready, so we only split on spaces
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary)

    X = vectorizer.fit_transform(docs)
    return X, vectorizer
