# Intro to Web Scraping

This module introduces practical web scraping with Python, covering static HTML, APIs, structured data, authenticated sessions, and JavaScript-rendered product pages.

## Why This Module Matters

Websites contain useful information that is often presented for people rather than directly consumable by software. Scraping techniques help turn permitted public or authenticated web content into structured data for analysis, monitoring, research, and automation. The module also demonstrates how to choose between direct HTTP requests, HTML parsing, API responses, and browser automation.

## What This Module Covers

- Fetching HTML reliably with `requests`, including status and timeout handling.
- Parsing static pages and following pagination with BeautifulSoup.
- Consuming paginated JSON APIs and extracting JSON-LD metadata.
- Preserving cookies and CSRF tokens during an authenticated request flow.
- Using Selenium and CSS selectors to collect static and detail-page product data.
- Scrolling through JavaScript-rendered content and preventing duplicate records.

The exercises are implemented in `0-fetch_html.py` through `8-scroll_and_scrape.py` and use quotes and product pages designed for scraping practice.

## Real Business Problems It Can Solve

With suitable authorization, patterns from this module can support:

- Competitive price, product availability, and catalog monitoring.
- Aggregating publicly available listings or research data into a common format.
- Collecting web content for market research, trend analysis, and reporting.
- Extracting structured metadata from publisher or commerce pages.
- Automating repetitive checks in authenticated internal portals.
- Building small data pipelines that detect new, changed, or duplicate records.

## Limitations in Business Settings

These exercises are intentionally lightweight and are not a complete production scraping platform. Real deployments must account for:

- Terms of service, copyright, privacy requirements, robots policies, and applicable law.
- Rate limits, anti-bot controls, CAPTCHAs, IP blocking, and changing access policies.
- Fragile HTML selectors, layout changes, missing fields, localization, and inconsistent data.
- Authentication security, secret management, authorization boundaries, and auditability.
- Browser-driver availability, higher infrastructure cost, and slower Selenium execution.
- Retry policies, observability, scheduling, storage, data validation, and quality monitoring.

Prefer an official API, data export, or a direct permission agreement when one is available. Scrape only data and pages you are authorized to access, use conservative request rates, and avoid collecting unnecessary personal information.

## Setup Workflow

Run the exercises from this directory with Python 3. The repository includes a local virtual environment named `.venv`; creating a fresh environment is also supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requests beautifulsoup4 selenium
```

Selenium requires a compatible Chrome or Chromium browser. Recent Selenium versions can manage the driver automatically; otherwise, install a matching ChromeDriver and make it available on `PATH`. The Selenium exercises are configured for headless Chrome at 1920x1080 without a sandbox.

## Suggested Workflow

1. Activate the environment and verify the required packages import successfully.
2. Start with `0-fetch_html.py`, checking HTTP errors, headers, and request timeouts.
3. Run the static quote exercises, then compare HTML pagination with the API approach.
4. Inspect a page's HTML and JSON-LD before writing or updating selectors.
5. Test the session-based login flow with authorized test credentials only.
6. Run the Selenium product exercises and validate fields such as title, price, description, and rating.
7. Test infinite scrolling with a bounded, respectful delay and verify duplicate handling.
8. Validate outputs, handle failures explicitly, and store results only when the use case permits it.

The scripts expose functions rather than a single command-line application, so each function can be imported and tested independently. Use the example URLs and credentials supplied by the exercise environment instead of embedding secrets in source code.

## Core Packages

- [`requests`](https://requests.readthedocs.io/): HTTP requests, APIs, and sessions.
- [`beautifulsoup4`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): HTML and JSON-LD extraction.
- [`selenium`](https://www.selenium.dev/documentation/): browser automation for dynamic pages.

## References

- [Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Requests documentation](https://requests.readthedocs.io/)
- [Selenium WebDriver documentation](https://www.selenium.dev/documentation/webdriver/)
- [MDN CSS selectors reference](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_selectors)
- [HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status)
- [RFC 6265: HTTP State Management Mechanism](https://www.rfc-editor.org/rfc/rfc6265)
