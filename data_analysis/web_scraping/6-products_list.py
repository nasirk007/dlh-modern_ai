#!/usr/bin/env python3
"""
Scrape products from a JS-rendered page with Selenium.
"""
import time
from selenium import webdriver


def scrape_products_list(url):
    """
    Open the page, wait for cards, and return product records.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        cards = []
        start = time.time()
        while time.time() - start < 10:
            cards = driver.find_elements("css selector", ".thumbnail")
            if cards:
                break
            time.sleep(0.25)

        products = []
        for card in cards:
            title = ""
            price = ""
            description = ""
            rating = 0

            title_el = card.find_elements("css selector", "a.title")
            if title_el:
                title = title_el[0].text.strip()

            price_el = card.find_elements("css selector", "h4.price")
            if price_el:
                price = price_el[0].text.strip()

            desc_el = card.find_elements("css selector", "p.description")
            if desc_el:
                description = desc_el[0].text.strip()

            rating_el = card.find_elements("css selector", "p[data-rating]")
            if rating_el:
                raw_rating = rating_el[0].get_attribute("data-rating") or "0"
                if raw_rating.isdigit():
                    rating = int(raw_rating)

            if rating == 0:
                stars = card.find_elements(
                    "css selector",
                    ".ws-icon-star, .glyphicon-star",
                )
                if not stars:
                    stars = card.find_elements(
                        "css selector",
                        ".ratings span",
                    )
                rating = len(stars)

            products.append(
                {
                    "title": str(title),
                    "price": str(price),
                    "description": str(description),
                    "rating": int(rating),
                }
            )

        return products
    finally:
        driver.quit()


def scrape_products(url):
    """
    Backward-compatible wrapper.
    """
    return scrape_products_list(url)


def products_list(url):
    """
    Backward-compatible wrapper.
    """
    return scrape_products_list(url)
