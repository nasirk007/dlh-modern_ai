#!/usr/bin/env python3
"""Build ngrams (groups of n tokens in a row) from a token list."""
import nltk


def generate_ngrams(tokens, n=2):
    """Make n-grams from the tokens and join each one with "_"
    (["call", "now"] -> "call_now"). Returns a list of strings."""
    if not isinstance(tokens, list) or len(tokens) < n:
        return []

    result = []
    for gram in nltk.ngrams(tokens, n):
        result.append("_".join(gram))
    return result
