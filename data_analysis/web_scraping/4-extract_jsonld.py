#!/usr/bin/env python3
"""
Extract quote data from embedded JSON-LD blocks.
"""
import json
from bs4 import BeautifulSoup

fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """
    Fetch a page, parse JSON-LD blocks, and return quote dicts.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    quotes_data = []

    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        raw = script.string or script.get_text()
        if not raw:
            continue

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue

            item_type = item.get("@type")
            is_quote = False
            if isinstance(item_type, str):
                is_quote = item_type == "Quote"
            elif isinstance(item_type, list):
                is_quote = "Quote" in item_type

            if not is_quote:
                continue

            text = item.get("text", "")

            author_data = item.get("author", {})
            if isinstance(author_data, dict):
                author = author_data.get("name", "")
            else:
                author = author_data or ""

            keywords = item.get("keywords", [])
            if isinstance(keywords, str):
                tags = [
                    tag.strip()
                    for tag in keywords.split(",")
                    if tag.strip()
                ]
            elif isinstance(keywords, list):
                tags = [
                    str(tag).strip()
                    for tag in keywords
                    if str(tag).strip()
                ]
            else:
                tags = []

            quotes_data.append({
                "text": text,
                "author": author,
                "tags": tags,
            })

    return quotes_data
