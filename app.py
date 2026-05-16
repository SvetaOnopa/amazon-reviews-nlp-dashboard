import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

from utils.preprocessing import load_data, filter_data

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #1f77b4; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }
    .section-title  { font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("amazon_reviews.csv")


df_full = get_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("Filters")

score_range = st.sidebar.slider("Star Rating", 1, 5, (1, 5))

versions_available = sorted(df_full["appVersion"].dropna().unique())
selected_versions = st.sidebar.multiselect(
    "App Version", versions_available, default=versions_available[:5] if len(versions_available) > 5 else versions_available
)

min_date = df_full["at"].dt.date.min()
max_date = df_full["at"].dt.date.max()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"]
)

search_term = st.sidebar.text_input("Search in Reviews", placeholder="keyword...")

# ── Apply filters ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = date_range
else:
    d0 = d1 = date_range[0]

df = filter_data(df_full, score_range, selected_versions, (d0, d1))
df = df[df["sentiment_label"].isin(sentiment_filter)]
if search_term:
    df = df[df["content"].str.contains(search_term, case=False, na=False)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📦 Amazon Reviews Dashboard")
st.caption(f"Showing **{len(df):,}** reviews out of {len(df_full):,} total")

if df.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_rating  = df["score"].mean()
avg_sent    = df["sentiment_score"].mean()
pct_pos     = (df["sentiment_label"] == "Positive").mean() * 100
top_upvoted = df["thumbsUpCount"].max()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h2>{avg_rating:.2f} ⭐</h2><p>Avg Star Rating</p></div>', unsafe_allow_html=True)
with c2:
    color = "🟢" if avg_sent > 0 else ("🔴" if avg_sent < 0 else "🟡")
    st.markdown(f'<div class="metric-card"><h2>{avg_sent:+.3f} {color}</h2><p>Avg Sentiment Score</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h2>{pct_pos:.1f}%</h2><p>Positive Reviews</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h2>{top_upvoted:,}</h2><p>Most Upvoted Review</p></div>', unsafe_allow_html=True)

st.divider()

# ── Row 1: Rating Distribution + Sentiment Distribution ───────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<p class="section-title">Rating Distribution</p>', unsafe_allow_html=True)
    rating_counts = df["score"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Stars", "Count"]
    rating_counts["Stars"] = rating_counts["Stars"].astype(int).astype(str) + " ⭐"
    fig_rating = px.bar(
        rating_counts, x="Stars", y="Count",
        color="Count",
        color_continuous_scale=["#d62728", "#ff7f0e", "#ffdd57", "#2ca02c", "#1f77b4"],
        text="Count",
    )
    fig_rating.update_traces(textposition="outside")
    fig_rating.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig_rating, use_container_width=True)

with col_right:
    st.markdown('<p class="section-title">Sentiment Breakdown</p>', unsafe_allow_html=True)
    sent_counts = df["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#2ca02c", "Neutral": "#ff7f0e", "Negative": "#d62728"}
    fig_sent = px.pie(
        sent_counts, names="Sentiment", values="Count",
        color="Sentiment", color_discrete_map=color_map,
        hole=0.45,
    )
    fig_sent.update_layout(margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig_sent, use_container_width=True)

# ── Row 2: Reviews Over Time + Avg Rating Over Time ───────────────────────────
st.markdown('<p class="section-title">Trends Over Time</p>', unsafe_allow_html=True)
col_tl, col_tr = st.columns(2)

monthly = df.groupby("month").agg(
    review_count=("reviewId", "count"),
    avg_rating=("score", "mean"),
    avg_sentiment=("sentiment_score", "mean"),
).reset_index().sort_values("month")

with col_tl:
    fig_vol = px.area(monthly, x="month", y="review_count", labels={"month": "Month", "review_count": "Reviews"})
    fig_vol.update_layout(margin=dict(t=10, b=10), height=280, xaxis_tickangle=-30)
    st.plotly_chart(fig_vol, use_container_width=True)

with col_tr:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_rating"], name="Avg Rating", line=dict(color="#1f77b4")))
    fig_trend.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_sentiment"], name="Avg Sentiment", line=dict(color="#2ca02c", dash="dash"), yaxis="y2"))
    fig_trend.update_layout(
        yaxis=dict(title="Avg Rating", range=[1, 5]),
        yaxis2=dict(title="Avg Sentiment", overlaying="y", side="right", range=[-1, 1]),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=10, b=10),
        height=280,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ── Row 3: Sentiment vs Rating scatter + Version comparison ───────────────────
col_sl, col_sr = st.columns(2)

with col_sl:
    st.markdown('<p class="section-title">Sentiment Score vs Star Rating</p>', unsafe_allow_html=True)
    fig_scatter = px.strip(
        df.sample(min(1000, len(df)), random_state=42),
        x="score", y="sentiment_score",
        color="sentiment_label",
        color_discrete_map=color_map,
        labels={"score": "Star Rating", "sentiment_score": "Sentiment Score"},
        hover_data=["userName"],
    )
    fig_scatter.update_layout(margin=dict(t=10, b=10), height=320, showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_sr:
    st.markdown('<p class="section-title">Avg Rating by App Version (Top 10)</p>', unsafe_allow_html=True)
    version_stats = (
        df.groupby("appVersion")
        .agg(avg_rating=("score", "mean"), count=("reviewId", "count"))
        .query("count >= 3")
        .sort_values("avg_rating", ascending=False)
        .head(10)
        .reset_index()
    )
    fig_ver = px.bar(
        version_stats, x="avg_rating", y="appVersion", orientation="h",
        color="avg_rating", color_continuous_scale="RdYlGn", range_color=[1, 5],
        text=version_stats["avg_rating"].round(2),
        labels={"appVersion": "Version", "avg_rating": "Avg Rating"},
    )
    fig_ver.update_traces(textposition="outside")
    fig_ver.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=320, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_ver, use_container_width=True)

# ── Row 4: Word Clouds ────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Word Cloud by Sentiment</p>', unsafe_allow_html=True)

wc_cols = st.columns(3)
wc_settings = [
    ("Positive", "#2ca02c", wc_cols[0]),
    ("Neutral",  "#ff7f0e", wc_cols[1]),
    ("Negative", "#d62728", wc_cols[2]),
]

STOPWORDS = {"the", "and", "is", "it", "to", "a", "of", "in", "for", "on", "with", "this", "that", "was", "are", "i", "my", "they", "have", "but"}

for label, color, col in wc_settings:
    subset = df[df["sentiment_label"] == label]["content"]
    text = " ".join(subset)
    with col:
        st.caption(f"**{label}** ({len(subset):,} reviews)")
        if len(text.strip()) < 10:
            st.info("Not enough text.")
        else:
            wc = WordCloud(
                width=400, height=250, background_color="white",
                colormap="Greens" if label == "Positive" else ("Oranges" if label == "Neutral" else "Reds"),
                stopwords=STOPWORDS, max_words=80, collocations=False,
            ).generate(text)
            fig_wc, ax = plt.subplots(figsize=(4, 2.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)

# ── Row 5: Top upvoted reviews ────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">Most Upvoted Reviews</p>', unsafe_allow_html=True)

top_reviews = (
    df[df["thumbsUpCount"] > 0]
    .sort_values("thumbsUpCount", ascending=False)
    .head(5)[["userName", "score", "thumbsUpCount", "sentiment_label", "content", "at"]]
    .reset_index(drop=True)
)
top_reviews.index += 1

for _, row in top_reviews.iterrows():
    stars = "⭐" * int(row["score"])
    badge_color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}.get(row["sentiment_label"], "gray")
    st.markdown(
        f'**{row["userName"]}** &nbsp; {stars} &nbsp; '
        f'<span style="color:{badge_color};font-weight:600">{row["sentiment_label"]}</span> &nbsp; '
        f'👍 {row["thumbsUpCount"]}',
        unsafe_allow_html=True,
    )
    st.caption(str(row["at"])[:10])
    st.write(f"> {row['content'][:300]}{'…' if len(row['content']) > 300 else ''}")
    st.divider()

# ── Row 6: Review Browser ─────────────────────────────────────────────────────
st.markdown('<p class="section-title">Browse Reviews</p>', unsafe_allow_html=True)

page_size = 20
total_pages = max(1, (len(df) - 1) // page_size + 1)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * page_size

display_cols = ["userName", "score", "sentiment_label", "sentiment_score", "thumbsUpCount", "appVersion", "at", "content"]
st.dataframe(
    df[display_cols]
    .sort_values("at", ascending=False)
    .iloc[start : start + page_size]
    .reset_index(drop=True),
    use_container_width=True,
    height=500,
    column_config={
        "score":           st.column_config.NumberColumn("⭐ Stars", format="%d"),
        "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
        "thumbsUpCount":   st.column_config.NumberColumn("👍 Upvotes"),
        "at":              st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
        "content":         st.column_config.TextColumn("Review", width="large"),
    },
)

st.caption(f"Page {page} of {total_pages} · {len(df):,} reviews")
