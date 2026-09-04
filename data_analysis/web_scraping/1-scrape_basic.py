#!/usr/bin/env python3
"""
Basic scaping with beautifulsoup
"""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """
    Next step after fetching HTML
    """

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    quotes_data = []
    for quote in soup.find_all("div", class_="quote"):
        text_tag = quote.find("span", class_="text")
        author_tag = quote.find("small", class_="author")
        tag_links = quote.select("div.tags a.tag")

        quotes_data.append({
            "text": text_tag.get_text(strip=True) if text_tag else "",
            "author": author_tag.get_text(strip=True) if author_tag else "",
            "tags": [tag.get_text(strip=True) for tag in tag_links],
        })

    return quotes_data
