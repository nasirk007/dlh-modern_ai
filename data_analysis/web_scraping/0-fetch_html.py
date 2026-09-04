#!/usr/bin/env python3
"""
Fetch a webpage and return HTML as text
"""
import requests


def fetch_html(url, headers=None, timeout=10):
    """
    Uses requests
    """

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text
