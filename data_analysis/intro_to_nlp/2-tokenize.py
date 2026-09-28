#!/usr/bin/env python3
"""Split cleaned SMS messages into tokens and handle ASCII emoticons
like :) or <3."""
import nltk

EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    """Replace emoticon tokens with <EMO>, or drop them when
    emoticon_action is not "replace". Returns a new list."""
    if not isinstance(tokens, list):
        return []
    result = []
    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())
        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)
    return result


def tokenize_text(text, method="tweet"):
    """Split a cleaned SMS message into tokens with the chosen method
    ("tweet", "word" or "split"). Returns a list of tokens."""
    # only strings can be tokenized
    if not isinstance(text, str):
        return []

    if method == "tweet":
        # reduce_len keeps at most 3 repeated characters
        tokenizer = nltk.tokenize.TweetTokenizer(reduce_len=True)
        tokens = tokenizer.tokenize(text)
    elif method == "word":
        tokens = nltk.word_tokenize(text)
    elif method == "split":
        tokens = text.split()
    else:
        raise ValueError("Invalid tokenizer method")

    return tokens
