#!/usr/bin/env python3
"""Learn FastText word vectors and turn each message into a vector."""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=50, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """Train FastText and return each message's mean token vector."""
    # Keep the corpus reusable for training and message-level pooling.
    sentences = list(corpus_tokens)

    # Learn word and character n-gram vectors from the tokenized messages.
    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   sg=sg,
                                   epochs=epochs,
                                   workers=workers)

    # Mean-pool each message; FastText also generates vectors for OOV tokens.
    rows = []
    for tokens in sentences:
        if tokens:
            rows.append(np.mean([model.wv[token] for token in tokens], axis=0))
        else:
            # An empty message has no token vectors to average.
            rows.append(np.zeros(vector_size))

    # Preserve a 2D shape, including when the corpus has no messages.
    X = np.asarray(rows, dtype=np.float64).reshape(-1, vector_size)
    return X, model
