"""URL feature extraction for phishing website detection.

All features are lexical / host-based and are computed from the URL string
alone (no network calls), so extraction is fast and safe.
"""
import math
import re
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "wallet", "suspend", "billing", "support", "alert",
    "free", "bonus", "gift", "recover", "unlock", "validate",
]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "support", "buzz", "info", "loan", "icu"}
BRANDS = ["paypal", "google", "apple", "amazon", "microsoft", "facebook", "netflix",
          "sbi", "hdfc", "icici", "paytm", "instagram", "whatsapp", "linkedin", "dropbox"]
TWO_PART_TLDS = {"co.uk", "co.in", "com.au", "co.jp", "com.br", "org.uk", "ac.in", "gov.in"}

IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_digits", "digit_ratio",
    "num_at", "num_question", "num_equals", "num_ampersand", "num_percent",
    "num_slashes", "path_depth", "num_subdomains",
    "has_ip_host", "has_https", "has_port", "has_punycode",
    "is_shortener", "suspicious_tld", "double_slash_in_path",
    "suspicious_word_count", "brand_in_subdomain_or_path", "host_hyphens",
    "url_entropy", "longest_token_length",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _split_host(host: str):
    """Return (subdomain, registered_domain, tld)."""
    parts = host.split(".")
    if len(parts) < 2:
        return "", host, ""
    last_two = ".".join(parts[-2:])
    if last_two in TWO_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[:-3]), parts[-3], last_two
    return ".".join(parts[:-2]), parts[-2], parts[-1]


def extract_features(url: str) -> dict:
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    subdomain, reg_domain, tld = _split_host(host)
    low = url.lower()

    digits = sum(ch.isdigit() for ch in url)
    tokens = re.split(r"[^a-zA-Z0-9]+", low)
    brand_hit = any(b in subdomain or b in path.lower() for b in BRANDS)
    # a brand in the registered domain itself (e.g. paypal.com) is not suspicious
    if any(b == reg_domain for b in BRANDS):
        brand_hit = any(b in subdomain or b in path.lower() for b in BRANDS)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "num_slashes": url.count("/"),
        "path_depth": len([p for p in path.split("/") if p]),
        "num_subdomains": len([s for s in subdomain.split(".") if s]) if subdomain else 0,
        "has_ip_host": int(bool(IP_RE.match(host))),
        "has_https": int(parsed.scheme == "https"),
        "has_port": int(parsed.port is not None) if _safe_port(parsed) else 0,
        "has_punycode": int("xn--" in host),
        "is_shortener": int(host in SHORTENERS),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "double_slash_in_path": int("//" in path),
        "suspicious_word_count": sum(low.count(w) for w in SUSPICIOUS_WORDS),
        "brand_in_subdomain_or_path": int(brand_hit),
        "host_hyphens": host.count("-"),
        "url_entropy": _entropy(url),
        "longest_token_length": max((len(t) for t in tokens), default=0),
    }


def _safe_port(parsed):
    try:
        return parsed.port
    except ValueError:
        return None


def build_feature_frame(urls) -> pd.DataFrame:
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
