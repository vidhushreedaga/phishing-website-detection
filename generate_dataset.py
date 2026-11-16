"""Generate a labelled URL dataset (url, label) -> data/urls.csv.

label 1 = phishing, label 0 = legitimate.

NOTE: This is a *synthetic* dataset built from realistic URL patterns so the
project runs offline. To use real data (PhishTank / OpenPhish for phishing,
Tranco / Alexa top sites for legitimate), just replace data/urls.csv with a CSV
that has the columns `url,label` and re-run train.py.
"""
import random
import string
from pathlib import Path

import pandas as pd

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "wikipedia.org", "github.com", "stackoverflow.com",
    "amazon.in", "flipkart.com", "nptel.ac.in", "iitb.ac.in", "bbc.co.uk",
    "nytimes.com", "linkedin.com", "microsoft.com", "apple.com", "python.org",
    "medium.com", "coursera.org", "kaggle.com", "paypal.com", "netflix.com",
    "hdfcbank.com", "onlinesbi.sbi", "icicibank.com", "paytm.com", "irctc.co.in",
    "reddit.com", "quora.com", "zomato.com", "swiggy.com", "ndtv.com",
    "thehindu.com", "imdb.com", "twitter.com", "instagram.com", "dropbox.com",
]
LEGIT_SUBS = ["www", "www", "www", "mail", "docs", "blog", "support", "en", "learn", "news", "m"]
LEGIT_PATHS = [
    "", "", "", "about", "contact", "products", "careers", "help/faq", "wiki/Machine_learning",
    "questions/12345/how-to-train-a-model", "courses/data-science", "news/2026/11/story",
    "docs/api/v2/reference", "login", "account/settings", "search", "category/electronics",
    "r/MachineLearning/comments/abc123", "user/profile", "pricing", "download", "support/billing",
]
LEGIT_QUERIES = ["", "", "", "", "?q=machine+learning", "?id=42&ref=home", "?page=3", "?lang=en&utm_source=mail"]

BRANDS = ["paypal", "google", "apple", "amazon", "microsoft", "facebook", "netflix",
          "sbi", "hdfc", "icici", "paytm", "instagram", "whatsapp", "linkedin"]
BAD_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "support", "buzz", "info", "icu", "com", "net", "online", "site"]
KEYWORDS = ["login", "signin", "verify", "update", "secure", "account", "confirm", "password",
            "banking", "wallet", "suspend", "billing", "alert", "recover", "unlock", "validate"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd"]


def rnd(n, alphabet=string.ascii_lowercase + string.digits):
    return "".join(random.choice(alphabet) for _ in range(n))


def legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    host = domain if random.random() < 0.4 else f"{random.choice(LEGIT_SUBS)}.{domain}"
    scheme = "https" if random.random() < 0.93 else "http"
    path = random.choice(LEGIT_PATHS)
    q = random.choice(LEGIT_QUERIES)
    url = f"{scheme}://{host}/{path}{q}"
    # a few long, messy-but-legitimate URLs (tracking links, article slugs)
    if random.random() < 0.08:
        url += ("&" if "?" in url else "?") + f"sessionid={rnd(24)}&utm_campaign={rnd(8)}"
    if random.random() < 0.05:
        url = f"https://{random.choice(LEGIT_SUBS)}.{domain}/{rnd(6)}-{rnd(5)}-{rnd(7)}/{random.randint(1000, 99999)}"
    return url


def phish_url():
    kind = random.choices(
        ["brand_sub", "ip", "keyword_domain", "shortener", "at_trick", "long_random", "typo", "subtle"],
        weights=[22, 10, 20, 6, 6, 12, 12, 12],
    )[0]
    brand = random.choice(BRANDS)
    kw = random.choice(KEYWORDS)
    tld = random.choice(BAD_TLDS)
    scheme = "https" if random.random() < 0.55 else "http"  # phishing sites use HTTPS too

    if kind == "brand_sub":
        url = f"{scheme}://{brand}.{kw}.{rnd(random.randint(5, 9))}.{tld}/{kw}/{rnd(8)}.php"
    elif kind == "ip":
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        url = f"http://{ip}/{brand}/{kw}/index.html"
    elif kind == "keyword_domain":
        url = f"{scheme}://{brand}-{kw}-{random.choice(KEYWORDS)}.{tld}/{kw}?id={rnd(10)}&session={rnd(16)}"
    elif kind == "shortener":
        url = f"https://{random.choice(SHORTENERS)}/{rnd(random.randint(5, 8), string.ascii_letters + string.digits)}"
    elif kind == "at_trick":
        url = f"{scheme}://www.{brand}.com@{rnd(8)}.{tld}/{kw}/"
    elif kind == "long_random":
        url = f"{scheme}://{rnd(random.randint(10, 18))}.{tld}/{rnd(12)}/{rnd(10)}/{kw}.html?token={rnd(32)}"
    elif kind == "typo":
        typo = brand.replace("o", "0").replace("l", "1").replace("a", "4", 1)
        url = f"{scheme}://{typo}.{random.choice(['com', 'net', tld])}/{kw}/{rnd(6)}"
    else:  # subtle: looks fairly normal, hard cases
        url = f"{scheme}://{brand}{random.randint(1, 99)}.{random.choice(['com', 'net', 'online', tld])}/{random.choice(['', kw, 'home'])}"
    if random.random() < 0.06:
        url = url.replace("://", "://xn--" + rnd(5) + ".", 1) if "@" not in url else url
    return url


def main(n_per_class: int = 3000, label_noise: float = 0.02):
    rows = [(legit_url(), 0) for _ in range(n_per_class)] + [(phish_url(), 1) for _ in range(n_per_class)]
    df = pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates("url")
    # small amount of label noise to mimic real-world mislabelled data
    flip = df.sample(frac=label_noise, random_state=1).index
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    out = Path(__file__).parent / "data" / "urls.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} URLs to {out}  (phishing share: {df.label.mean():.2%})")


if __name__ == "__main__":
    main()
