# Phishing Website Detection using URL Features (Random Forest vs SVM)

A machine learning mini project that classifies a URL as **phishing** or **legitimate**
using only features extracted from the URL string (no page content, no network calls).

## 1. Problem
Phishing sites imitate trusted brands to steal credentials. Blacklists are slow to catch new
sites, so ML models that judge a URL by its structure can flag suspicious links earlier.

## 2. Project structure
```
phishing_detection/
├── generate_dataset.py   # builds data/urls.csv (url,label)
├── features.py           # 29 URL features + extractor
├── train.py              # trains/tunes Random Forest & SVM, saves plots + metrics
├── predict.py            # CLI to classify any URL
├── requirements.txt
├── data/                 # urls.csv, features.csv
├── models/               # random.joblib, svm.joblib
└── outputs/              # confusion matrices, ROC, comparison, feature importance, metrics
```

## 3. How to run
```bash
pip install -r requirements.txt
python generate_dataset.py     # (skip if you bring your own data)
python train.py                # trains both models, writes outputs/
python predict.py "http://paypal.secure-login.verify-account.xyz/signin/x8d7f.php" https://www.wikipedia.org
```

## 4. Features (29)
| Group | Features |
|---|---|
| Length | URL, host, path, query length |
| Character counts | dots, hyphens, underscores, digits, digit ratio, `@`, `?`, `=`, `&`, `%`, `/` |
| Structure | path depth, number of subdomains, longest token, `//` inside path |
| Host tricks | IP address as host, port present, punycode (`xn--`), URL shortener, suspicious TLD (.xyz, .tk ...) |
| Semantics | count of words like *login, verify, secure*; brand name in subdomain/path (e.g. `paypal.evil.xyz`) |
| Statistical | Shannon entropy of the URL (random-looking strings score high) |
| Protocol | HTTPS used (not a reliable signal alone: many phishing sites use HTTPS) |

## 5. Method
1. Extract features from each URL.
2. Stratified 80/20 train/test split.
3. **Random Forest**: grid search over trees, depth, min leaf size (5-fold CV, F1).
4. **SVM (RBF kernel)**: `StandardScaler` + SVC, grid search over C and gamma (5-fold CV, F1).
   Scaling is essential for SVM because features have very different ranges; tree models don't need it.
5. Evaluate on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

## 6. Results (held-out test set, 1150 URLs)
| Metric | Random Forest | SVM (RBF) |
|---|---|---|
| Accuracy | 0.9739 | 0.9739 |
| Precision | 0.9782 | 0.9782 |
| Recall | 0.9717 | 0.9717 |
| F1 | 0.9750 | 0.9750 |
| ROC-AUC | 0.9740 | 0.9742 |

Plots are in `outputs/` (`roc_curves.png`, `model_comparison.png`, `feature_importance.png`,
`confusion_random.png`, `confusion_svm.png`). Exact numbers are also in `outputs/metrics.csv`.

## 7. IMPORTANT: about the dataset
The included `data/urls.csv` is **synthetic**: it was generated from realistic phishing and
legitimate URL patterns (with ~2% label noise) so the project runs offline. Because the
generator and the features share the same patterns, scores here are optimistic and **will be
lower on real-world data**. For your submission, you can (and should) swap in a real dataset:

- Phishing URLs: PhishTank or OpenPhish feeds
- Legitimate URLs: Tranco top-sites list
- Or the UCI / Kaggle "Phishing Websites" datasets (for URL-only data, keep just the URL column)

Create a CSV with columns `url,label` (1 = phishing, 0 = legitimate) and run
`python train.py --data your_file.csv`. Everything else works unchanged.

## 8. Limitations and future work
- Only lexical features; adding WHOIS domain age, DNS records, SSL certificate details and page
  content would improve accuracy.
- Attackers adapt, so models need periodic retraining.
- Try Gradient Boosting / XGBoost, character-level CNN on raw URLs, and SMOTE for imbalanced data.

## 9. Likely viva questions
- **Why scale features for SVM but not Random Forest?** SVM uses distances; trees split on thresholds.
- **Why F1 instead of accuracy?** It balances precision and recall; missing a phishing URL (false negative) is costly.
- **Why is HTTPS not enough?** Free certificates mean many phishing sites use HTTPS.
- **What does the RBF kernel do?** Maps data to a higher-dimensional space to separate non-linear classes.
