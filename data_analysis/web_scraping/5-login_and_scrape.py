#!/usr/bin/env python3
"""
Log in to quotes.toscrape.com and scrape authenticated quote data.
"""
import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """
    Log in with CSRF token, then scrape quotes from the protected page.
    """
    with requests.Session() as session:
        get_resp = session.get(login_url, timeout=10)
        get_resp.raise_for_status()

        soup = BeautifulSoup(get_resp.text, "html.parser")
        token_input = soup.find("input", attrs={"name": "csrf_token"})
        csrf_token = token_input["value"]

        payload = {
            "username": user,
            "password": pwd,
            "csrf_token": csrf_token,
        }

        post_resp = session.post(login_url, data=payload, timeout=10)
        post_resp.raise_for_status()

        if login_url.endswith("/login"):
            protected_url = login_url[:-6] + "/"
        elif login_url.endswith("/login/"):
            protected_url = login_url[:-7] + "/"
        else:
            protected_url = "https://quotes.toscrape.com/"

        page_resp = session.get(protected_url, timeout=10)
        page_resp.raise_for_status()

        page_soup = BeautifulSoup(page_resp.text, "html.parser")
        quotes_data = []

        for quote in page_soup.find_all("div", class_="quote"):
            text_tag = quote.find("span", class_="text")
            author_tag = quote.find("small", class_="author")
            tag_links = quote.select("div.tags a.tag")

            quotes_data.append({
                "text": text_tag.get_text(strip=True) if text_tag else "",
                "author": (
                    author_tag.get_text(strip=True) if author_tag else ""
                    ),
                "tags": [tag.get_text(strip=True) for tag in tag_links],
            })

        return quotes_data
