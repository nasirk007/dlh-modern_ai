#!/usr/bin/env python3
"""Throw away tokens that carry little meaning, like single letters or
pure numbers, but keep pipeline placeholders like <NUM>."""
import re

_PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """Keep only useful tokens: placeholders always stay, short tokens
    and tokens without letters are dropped. Returns a new list."""
    if not tokens:
        return []

    result = []
    for token in tokens:
        # placeholders like <NUM> always stay
        if _PLACEHOLDER_RE.match(token):
            result.append(token)
            continue

        # "#free" -> "free"
        if strip_hashtag and token.startswith("#"):
            token = token[1:]

        # too short
        if len(token) < min_len:
            continue

        # no letter at all (like "..." or "18")
        has_letter = False
        for char in token:
            if char.isalpha():
                has_letter = True
        if not has_letter:
            continue

        result.append(token)
    return result
