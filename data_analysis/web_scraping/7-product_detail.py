#!/usr/bin/env python3
"""
Scrape a single product detail page using Selenium.
"""
import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """
    Open one product page and return title, price, description, and rating.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(delay)

        title = ""
        price = ""
        description = ""
        rating = 0

        title_nodes = driver.find_elements("css selector", ".caption h4")
        if len(title_nodes) >= 2:
            title = title_nodes[1].text.strip()

        price_nodes = driver.find_elements("css selector", "h4.price")
        if price_nodes:
            price = price_nodes[0].text.strip()

        desc_nodes = driver.find_elements("css selector", "p.description")
        if desc_nodes:
            description = desc_nodes[0].text.strip()

        stars = []
        rating_nodes = driver.find_elements("css selector", "p[data-rating]")
        if rating_nodes:
            raw_rating = rating_nodes[0].get_attribute("data-rating") or "0"
            if raw_rating.isdigit():
                rating = int(raw_rating)

        rating = 0
        rating_nodes = driver.find_elements("css selector", "[data-rating]")
        if rating_nodes:
            raw_rating = rating_nodes[0].get_attribute("data-rating") or ""
            digits = "".join(ch for ch in raw_rating if ch.isdigit())
            if digits:
                rating = int(digits)

        if rating == 0:
            stars = driver.find_elements(
                "css selector",
                ".ratings .ws-icon-star",
            )
            if not stars:
                stars = driver.find_elements(
                    "css selector",
                    ".ratings .glyphicon-star",
                )
            if not stars:
                stars = driver.find_elements(
                    "css selector",
                    ".ratings span",
                )
            rating = len(stars)

        return {
            "title": str(title),
            "price": str(price),
            "description": str(description),
            "rating": int(rating),
        }
    finally:
        driver.quit()
