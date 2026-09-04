#!/usr/bin/env python3
"""
Scroll a dynamic products page and scrape unique products with Selenium.
"""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=0.2):
    """
    Scroll until loaded cards stabilize, then return unique product records.
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

        stable_rounds = 0
        last_count = 0
        start = time.time()
        max_wait = 14.0
        max_steps = 120

        for _ in range(max_steps):
            if time.time() - start > max_wait:
                break

            current_count = driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
                "return document.querySelectorAll('div.thumbnail').length;"
            )

            if current_count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_count = current_count

            if stable_rounds >= 10:
                break

            time.sleep(scroll_pause)

        raw_products = driver.execute_script(
            "const q='div.thumbnail';"
            "const cards=Array.from(document.querySelectorAll(q));"
            "return cards.map((card)=>{"
            "const t=card.querySelector('a.title');"
            "const p=card.querySelector('h4.price');"
            "const d=card.querySelector('p.description');"
            "const s1='.ratings .ws-icon-star';"
            "const s2='.ratings .glyphicon-star';"
            "const stars=card.querySelectorAll(s1+', '+s2).length;"
            "const title=t?(t.getAttribute('title')||t.textContent).trim():'';"
            "const price=p?p.textContent.trim():'';"
            "const description=d?d.textContent.trim():'';"
            "return {title:title,price:price,description:description,"
            "rating:stars};"
            "});"
        )

        products = []
        seen = set()

        for item in raw_products:
            title = str(item.get("title", ""))
            price = str(item.get("price", ""))
            description = str(item.get("description", ""))
            rating = int(item.get("rating", 0))

            key = (title, price)
            if key in seen:
                continue
            seen.add(key)

            products.append(
                {
                    "title": title,
                    "price": price,
                    "description": description,
                    "rating": rating,
                }
            )

        return products
    finally:
        driver.quit()
