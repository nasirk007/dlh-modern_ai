#!/usr/bin/env python3
"""Reduce words to a base-form with lemmatization or stemming."""
import nltk
import re


def get_pos(tag):
    """Turn Pen Treebank tag from nltk.pos_tag() (like 'VBD' or 'JJ')
    into WordNet POS that the lemmatizer understands."""
    if tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    if tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    if tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN


def normalize_tokens(tokens, method="lemmatize"):
    """Lemmatize or stem each token, leaving placeholders like <NUM>
    untouched. Returns the new list of tokens."""
    if method != "lemmatize" and method != "stem":
        raise ValueError("method must be 'lemmatize' or 'stem'")

    result = []

    if method == "stem":
        stemmer = nltk.stem.PorterStemmer()
        for token in tokens:
            # do not touch placeholders like <NUM>
            if re.match(r'^<[A-Za-z]+>$', token):
                result.append(token)
            else:
                result.append(stemmer.stem(token))
        return result

    # lemmatize: first find the POS of each token
    tagged = nltk.pos_tag(tokens)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for token, tag in tagged:
        if re.match(r'^<[A-Za-z]+>$', token):
            result.append(token)
        else:
            pos = get_pos(tag)
            result.append(lemmatizer.lemmatize(token, pos=pos))
    return result
