import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def load_data(path: str = "amazon_reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["date"] = df["at"].dt.date
    df["month"] = df["at"].dt.to_period("M").astype(str)
    df["year"] = df["at"].dt.year
    df["content"] = df["content"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["score", "at"])
    df = _add_sentiment(df)
    df = _add_rating_label(df)
    return df


def _add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    scores = df["content"].apply(lambda t: _analyzer.polarity_scores(t)["compound"])
    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = df["sentiment_score"].apply(_compound_to_label)
    return df


def _compound_to_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def _add_rating_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating_label"] = df["score"].apply(
        lambda s: "Positive" if s >= 4 else ("Negative" if s <= 2 else "Neutral")
    )
    return df


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
