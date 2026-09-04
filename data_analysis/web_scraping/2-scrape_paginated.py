#!/usr/bin/env python3
"""Deep Dive to Statics"""
from bs4 import BeautifulSoup
import time
from urllib import parse

fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """Follows 'Next' links to the bottom"""
    all_quotes = []
    current_url = base_url

    while current_url:
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")

        for quote in soup.find_all("div", class_="quote"):
            text_tag = quote.find("span", class_="text")
            athr_tag = quote.find("small", class_="author")
            tag_links = quote.select("div.tags a.tag")

            all_quotes.append({
                "text": text_tag.get_text(strip=True) if text_tag else "",
                "author": athr_tag.get_text(strip=True) if athr_tag else "",
                "tags": [tag.get_text(strip=True) for tag in tag_links],
            })

        next_li = soup.find("li", class_="next")
        if next_li and next_li.find("a"):
            next_href = next_li.find("a").get("href")
            current_url = parse.urljoin(current_url, next_href)
            time.sleep(1)
        else:
            current_url = None

    return all_quotes
