#!/usr/bin/env python3
"""Learn Word2Vec word vectors and turn each message into one vector."""
import numpy as np
import gensim.models


def word2vec_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=2, sg=0, epochs=10, workers=4):
    """Train Word2Vec on the corpus and represent each message as the
    mean of its word vectors. Returns (X, model)."""
    # train the model
    model = gensim.models.Word2Vec(sentences=corpus_tokens,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   sg=sg,
                                   epochs=epochs,
                                   workers=workers)

    rows = []
    for tokens in corpus_tokens:
        # vectors of the words the model knows
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])

        # mean of the vectors, or zeros if there is none
        if len(vectors) > 0:
            rows.append(np.mean(vectors, axis=0))
        else:
            rows.append(np.zeros(vector_size))

    # gensim vectors are float32, the task wants float64
    X = np.array(rows, dtype=np.float64)
    return X, model
