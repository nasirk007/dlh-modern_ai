#!/usr/bin/env python3
"""Remove common filler words (stopwords) from a list of tokens."""
import nltk


def remove_stopwords(tokens, language="english", extra_words=None,
                     keep_words=None):
    """Drop stopwords from a token list. Words in extra_words are dropped
    too, words in keep_words are always kept. Returns the new list."""
    if not isinstance(tokens, list):
        return []

    # NLTK stopword list as a set
    stop_words = set(nltk.corpus.stopwords.words(language))

    # add the extra words
    if extra_words:
        for word in extra_words:
            stop_words.add(word.lower())

    # take out the words we want to keep
    if keep_words:
        for word in keep_words:
            stop_words.discard(word.lower())

    # keep only the tokens that are not stopwords
    # compare in lowercase, so "IS" is removed like "is"
    result = []
    for token in tokens:
        if token.lower() not in stop_words:
            result.append(token)
    return result
