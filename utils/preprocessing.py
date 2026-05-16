import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

_analyzer = SentimentIntensityAnalyzer()

COMPLAINT_CATEGORIES = {
    "Delivery Issues":     r"deliver|ship|arriv|late|package|track|dispatch",
    "Missing Orders":      r"not receiv|never receiv|missing order|lost order|never arriv|didn.t arriv|not arriv|no order",
    "Refund / Returns":    r"refund|return|money back|charg|billing|stolen|fraud|scam",
    "App Performance":     r"slow|crash|freez|bug|glitch|load|not working|broken|error|lag|stuck",
    "Customer Service":    r"customer service|support|help|respon|reply|agent|representative|contact",
    "Product Quality":     r"quality|fake|counterfeit|defective|broken product|damaged|third.party|third party|mislead",
    "Dark Mode / UI":      r"dark mode|dark background|dark theme|night mode|white background|glaring",
    "Payment Options":     r"payment|tamara|tabby|card|debit|credit|checkout|pay option",
    "Account / Login":     r"account|login|password|verif|sign in|log in|locked out",
    "Prime / Subscript.":  r"prime|membership|subscription|cancel prime|renew|trial",
    "Search & Filters":    r"search|filter|algorithm|sort|results|find product",
    "Pricing / Fees":      r"price|cost|fee|expensive|overpriced|delivery fee|hidden",
    "AI / Ads":            r"\bai\b|artificial intelligence|alexa|chatbot|advertis|sponsor|ad ",
}

POSITIVE_THEMES = {
    "Selection & Variety": r"selection|variety|choice|wide range|everything|product range",
    "Price & Value":       r"price|cheap|affordable|deal|value|discount|offer|bargain|save",
    "Fast Delivery":       r"fast deliver|quick deliver|same day|next day|on time|prompt",
    "Easy to Use":         r"easy|simple|user.friendly|intuitive|smooth|convenient|quick",
    "Prime Benefits":      r"prime|membership|benefit|free deliver|free ship",
    "Customer Service":    r"great service|excellent service|helpful|amazing support|good support",
    "Product Quality":     r"good quality|high quality|excellent product|great product|original",
}


def load_data(path: str = "amazon_reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["date"] = df["at"].dt.date
    df["month"] = df["at"].dt.to_period("M").astype(str)
    df["year"] = df["at"].dt.year.astype("Int64")
    df["quarter"] = df["at"].dt.to_period("Q").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0).astype(int)
    df["review_length"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len()
    df = df.dropna(subset=["score", "at"])
    df = _add_sentiment(df)
    df = _add_labels(df)
    return df


def _add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["compound"] = df["content"].apply(lambda t: _analyzer.polarity_scores(t)["compound"])
    df["sentiment"] = df["compound"].apply(
        lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
    )
    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating_label"] = df["score"].apply(
        lambda s: "Positive" if s >= 4 else ("Negative" if s <= 2 else "Neutral")
    )
    return df


def classify_complaint(text: str) -> list[str]:
    text = text.lower()
    return [cat for cat, pat in COMPLAINT_CATEGORIES.items() if re.search(pat, text)]


def classify_positive(text: str) -> list[str]:
    text = text.lower()
    return [cat for cat, pat in POSITIVE_THEMES.items() if re.search(pat, text)]


def get_top_keywords(texts: pd.Series, n: int = 20, ngram_range: tuple = (1, 2)) -> pd.DataFrame:
    clean = texts[texts.str.len() > 5]
    if len(clean) < 5:
        return pd.DataFrame(columns=["keyword", "score"])
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(clean)
    scores = tfidf.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_idx = scores.argsort()[-n:][::-1]
    return pd.DataFrame({"keyword": vocab[top_idx], "score": scores[top_idx]})


def filter_data(
    df: pd.DataFrame,
    score_range: tuple,
    versions: list,
    date_range: tuple,
) -> pd.DataFrame:
    mask = (
        df["score"].between(score_range[0], score_range[1])
        & df["at"].dt.date.between(date_range[0], date_range[1])
    )
    if versions:
        mask &= df["appVersion"].isin(versions)
    return df[mask]
