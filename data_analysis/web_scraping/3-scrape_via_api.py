#!/usr/bin/env python3
"""Fetch quote from API"""
import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """Quote fetch"""
    quotes_data = []
    page = 1
    root = base_url.rstrip("/")

    while True:
        url = f"{root}/api/quotes?page={page}"
        payload = json.loads(fetch_html(url))

        for quote in payload.get("quotes", []):
            author = quote.get("author", {})
            if isinstance(author, dict):
                author_name = author.get("name", "")
            else:
                author_name = author or ""

            quotes_data.append({
                "text": quote.get("text", ""),
                "author": author_name,
                "tags": quote.get("tags", []),
            })

        if not payload.get("has_next"):
            break
        page += 1

    return quotes_data
