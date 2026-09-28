#!/usr/bin/env python3
"""Clean raw SMS messages so the tokenizer gets tidy, lowercase text
with simple placeholders instead of URLs, numbers and emoji."""
import re
import emoji

_DATASET_PLACEHOLDER_MAP = {
    '<#>':       '<NUM>',
    '<decimal>': '<NUM>',
    '<time>':    '<TIME>',
    '<url>':     '<URL>',
    '<email>':   '<EMAIL>',
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        r"[‐‑‒–—―−]": "-",
        r"…":          "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """Clean one SMS message: lowercase it and replace URLs, numbers,
    emoji and noisy punctuation. Returns the cleaned string."""
    # 1. lowercase + strip
    if text is None:
        return ""
    text = text.lower()
    text = text.strip()

    # 2. dataset placeholders
    for key in _DATASET_PLACEHOLDER_MAP:
        text = text.replace(key, _DATASET_PLACEHOLDER_MAP[key])

    # 3. curly quotes, dashes, ... to ASCII
    text = normalize_unicode_punct(text)

    # 4. URL replacement
    if replace_url:
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # 5. number replacement, phone numbers first
    if replace_num:
        text = re.sub(r'\+?\d[\d\s\-]{6,}\d', '<NUM>', text)
        pattern = r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b'
        text = re.sub(pattern, '<NUM>', text)

    # 6. emoji handling
    if emoji_action == "replace":
        text = emoji.replace_emoji(text, replace='<EMO>')
    elif emoji_action == "remove":
        text = emoji.replace_emoji(text, replace=' ')

    # 7. "!!!" -> "!" and "??" -> "?"
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    # 8. one space between words
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
