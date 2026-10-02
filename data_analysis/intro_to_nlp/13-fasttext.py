#!/usr/bin/env python3
"""Learn FastText word vectors and turn each message into a vector."""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=50, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """Train FastText and return each message's mean token vector."""
    sentences = list(corpus_tokens)
    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   sg=sg,
                                   epochs=epochs,
                                   workers=workers)

    rows = []
    for tokens in sentences:
        if tokens:
            rows.append(np.mean([model.wv[token] for token in tokens], axis=0))
        else:
            rows.append(np.zeros(vector_size))

    X = np.asarray(rows, dtype=np.float64).reshape(-1, vector_size)
    return X, model
