#!/usr/bin/env python3
"""Turn token lists into TF-IDF matrix, where words that are common
everywhere get lower weight."""
import sklearn


def tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2),
           min_df=2, max_df=0.95, norm='l2'):
    """Build TF-IDF matrix with TfidfVectorizer.
    Returns (X, vectorizer): the sparse matrix & the fitted vectorizer."""
    # each token list becomes one string
    docs = []
    for tokens in corpus_tokens:
        docs.append(" ".join(tokens))

    # the tokens are ready, so we only split on spaces
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm)

    # Learn vocabulary weights and encode documents as a sparse matrix.
    X = vectorizer.fit_transform(docs)
    return X, vectorizer
