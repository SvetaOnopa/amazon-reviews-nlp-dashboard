import os
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

_analyzer = SentimentIntensityAnalyzer()

SENTIMENT_CACHE = "analysis/sentiment_cache.parquet"

# ── Complaint / positive topic patterns ──────────────────────────────────────
KEYWORD_TOPICS = {
    # Delivery
    "📦 Late / Delayed Delivery":      r"late|delayed|delay|slow delivery|not on time|overdue|too long to arriv",
    "📦 Missing / Lost Order":          r"not receiv|never receiv|missing order|lost order|never arriv|didn.t arriv|where is my order|not arriv",
    "📦 Wrong Item Delivered":          r"wrong item|wrong product|incorrect item|different item|not what i ordered|sent wrong",
    "📦 Free Delivery Not Applied":     r"free delivery|free shipping|delivery option|no free|no delivery option",
    "🚚 Delivery Fee / Shipping Cost":  r"shipping fee|delivery fee|delivery charge|shipping cost|R\d+.*delivery|delivery.*cost",
    # Returns & Money
    "💸 Refund Not Received":           r"no refund|won.?t refund|refused refund|refund denied|refund not|can.?t get.*refund|never got.*refund",
    "🔄 Return Problems":               r"can.?t return|return policy|return process|return rejected|send back|return label",
    "💳 Charged / Billing Error":       r"charged|overcharged|double charge|unauthorized charge|billing error|charge me|charged again",
    "🔐 Fraud / Scam":                  r"fraud|scam|stolen|hack|unauthorized|scammer|cheat|lie|lies|lied",
    # App issues
    "🐛 App Crash / Freezing":          r"crash|freez|stuck|not respond|force close|keeps closing|app stop|keeps crash",
    "🐢 App Slow / Not Loading":        r"\bslow\b|lag|not loading|takes forever|loading forever|loads slow|very slow",
    "🔧 App Bug / Not Working":         r"\bbug\b|error|glitch|not working|broken|doesn.?t work|won.?t open|doesn.?t open|app broken",
    "🔍 Search or Filter Broken":       r"search (doesn.?t|not work|broken|wrong|fail)|filter (not|broken|doesn.?t|removed|gone)",
    # Account
    "👤 Login / Account Issue":         r"can.?t login|can.?t sign|account issue|login problem|log in|sign in|locked out",
    "🔑 Password / Verification":       r"password|verif|otp|verification code|two.factor|reset password",
    # Subscription
    "👑 Prime – Can't Cancel":          r"can.?t cancel prime|cancel prime|cancel membership|cancel subscription|auto.?renew|renew prime",
    "👑 Prime – Not Worth It":          r"prime not worth|prime expensive|prime useless|prime disappoint|prime waste|prime overpriced",
    # Customer Service
    "👨‍💼 Poor Customer Service":        r"customer service|customer support|support team|unhelpful|rude agent|bad service|terrible service",
    "📞 No Response from Support":      r"no response|never respond|ignor|no reply|unreachable|can.?t contact|no one help",
    # Products
    "🎭 Fake / Counterfeit Products":   r"fake|counterfeit|not original|not genuine|replica|knock.?off|not authentic",
    "📉 Bad Product Quality":           r"bad quality|poor quality|cheap quality|defective|damaged|not durable|fell apart|broke after",
    # UI / Features
    "🌙 Dark Mode Request":             r"dark mode|dark background|dark theme|night mode|white background|glaring|bright background",
    "💳 Payment Options Missing":       r"tamara|tabby|payment option|payment method|can.?t pay|checkout fail|no payment",
    "🤖 Unwanted AI / Alexa":           r"\bai\b|artificial intelligence|alexa|chatbot|ai feature|ai suggest|ai shopping",
    "📢 Too Many Ads":                  r"\bads\b|advertisement|sponsored|too many ad|ad everywhere|full of ads|ad clutter",
    "👗 Clothing Size Filter":          r"size filter|plus size|\b1[Xx]\b|clothing filter|can.?t filter.*size|removed.*filter|filter.*removed",
    # Pricing
    "💰 Overpriced / Hidden Fees":      r"expensive|overpriced|hidden fee|extra charge|too expensive|price increase|jacked up",
    # Positive topics
    "⭐ Great Product Selection":       r"great selection|huge selection|find anything|wide range|good variety|everything.*find|good selection",
    "💚 Good Prices / Deals":           r"great price|good price|affordable|cheap|great deal|discount|save money|best price|low price",
    "🚀 Fast / On-Time Delivery":       r"fast deliver|quick deliver|same day|next day|arrived fast|on time|speedy|delivery was fast",
    "✅ Easy to Use":                   r"easy to use|user.?friendly|convenient|simple|smooth|intuitive|easy.?to.?navigate",
    "❤️ Loves Amazon":                  r"love amazon|best app|amazing app|excellent app|wonderful|fantastic|perfect app|great app",
    "👑 Happy Prime Member":            r"prime is worth|love prime|prime benefit|prime value|prime member|prime advantage",
    "👍 Great Customer Service":        r"great service|excellent service|helpful.*support|amazing.*support|good.*customer service|fast.*response",
}

COMPLAINT_CATEGORIES = {
    "Delivery Issues":     r"deliver|ship|arriv|late|package|track|dispatch",
    "Missing Orders":      r"not receiv|never receiv|missing order|lost order|never arriv|not arriv",
    "Refund / Returns":    r"refund|return|money back|charg|billing|stolen|fraud|scam",
    "App Performance":     r"slow|crash|freez|bug|glitch|load|not working|broken|error|lag|stuck",
    "Customer Service":    r"customer service|support|help|respon|reply|agent|represent",
    "Product Quality":     r"quality|fake|counterfeit|defective|damaged|third.party|mislead",
    "Dark Mode / UI":      r"dark mode|dark background|dark theme|night mode|white background|glaring",
    "Payment Options":     r"payment|tamara|tabby|card|checkout|pay option",
    "Account / Login":     r"account|login|password|verif|sign in|log in|locked",
    "Prime / Subscript.":  r"prime|membership|subscription|cancel prime|renew|trial",
    "Search & Filters":    r"search|filter|algorithm|sort|results|find product",
    "Pricing / Fees":      r"price|fee|expensive|overpriced|delivery fee|hidden",
    "AI / Ads":            r"\bai\b|artificial intelligence|alexa|chatbot|advertis|sponsor",
}

POSITIVE_THEMES = {
    "Selection & Variety": r"selection|variety|choice|wide range|everything|find anything",
    "Price & Value":       r"price|cheap|afford|deal|value|discount|offer|bargain|save",
    "Fast Delivery":       r"fast deliver|quick deliver|same day|next day|on time|prompt",
    "Easy to Use":         r"easy|simple|user.friend|intuitive|smooth|convenient",
    "Prime Benefits":      r"prime|membership|benefit|free deliver|free ship",
    "Customer Service":    r"great service|excellent service|helpful|amazing support|good support",
    "Product Quality":     r"good quality|high quality|excellent product|great product|original",
}


def load_data(path: str = "amazon_reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["date"]    = df["at"].dt.date
    df["month"]   = df["at"].dt.to_period("M").astype(str)
    df["year"]    = df["at"].dt.year.astype("Int64")
    df["quarter"] = df["at"].dt.to_period("Q").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["score"]   = pd.to_numeric(df["score"], errors="coerce")
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0).astype(int)
    df["review_length"] = df["content"].str.len()
    df["word_count"]    = df["content"].str.split().str.len()
    df = df.dropna(subset=["score", "at"])
    df = _add_sentiment(df)
    df = _add_labels(df)
    return df


def _add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if os.path.exists(SENTIMENT_CACHE):
        cache = pd.read_parquet(SENTIMENT_CACHE)[["reviewId", "compound", "sentiment"]]
        df = df.merge(cache, on="reviewId", how="left")
        # Fall back to VADER for any rows missing from cache
        missing = df["compound"].isna()
        if missing.any():
            df.loc[missing, "compound"] = df.loc[missing, "content"].apply(
                lambda t: _analyzer.polarity_scores(t)["compound"]
            )
            df.loc[missing, "sentiment"] = df.loc[missing, "compound"].apply(_compound_to_label)
    else:
        df["compound"] = df["content"].apply(lambda t: _analyzer.polarity_scores(t)["compound"])
        df["sentiment"] = df["compound"].apply(_compound_to_label)
    return df


def _compound_to_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


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
        stop_words="english", max_features=5000,
        ngram_range=ngram_range, min_df=2, sublinear_tf=True,
    )
    tfidf = vec.fit_transform(clean)
    scores = tfidf.sum(axis=0).A1
    vocab  = vec.get_feature_names_out()
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
