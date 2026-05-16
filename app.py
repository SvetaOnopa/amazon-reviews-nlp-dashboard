import io
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from utils.preprocessing import (
    load_data, filter_data, get_top_keywords,
    COMPLAINT_CATEGORIES, POSITIVE_THEMES,
    classify_complaint, classify_positive,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
.kpi-label { font-size: 0.78rem; color: #888; margin-top: -6px; }
.section-header {
    font-size: 1.05rem; font-weight: 700;
    border-left: 4px solid #1f77b4; padding-left: 10px;
    margin: 18px 0 8px 0;
}
.insight-box {
    background: #f7f9fc; border-left: 4px solid #ff7f0e;
    border-radius: 4px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.92rem;
}
.tag-pos { background:#e6f4ea; color:#1a7c34; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
.tag-neg { background:#fce8e6; color:#c5221f; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
.tag-neu { background:#fff3e0; color:#b45309; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

PALETTE = {"Positive": "#2ca02c", "Neutral": "#ff7f0e", "Negative": "#d62728"}
RATING_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#ffdd57", 4: "#98df8a", 5: "#2ca02c"}
EXTRA_STOPS = STOPWORDS | {"amazon", "app", "use", "one", "get", "now", "just", "like",
                             "would", "got", "even", "know", "really", "much", "time",
                             "still", "also", "used", "make", "give", "ve", "don", "ll"}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & scoring 87K reviews…")
def get_data():
    return load_data("amazon_reviews.csv")

df_full = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📦 Amazon Reviews")
    st.caption("87,112 reviews · 2018–2026")
    st.divider()

    st.subheader("Filters")
    score_range = st.slider("⭐ Star Rating", 1, 5, (1, 5))

    min_date = df_full["at"].dt.date.min()
    max_date = df_full["at"].dt.date.max()
    date_range = st.date_input("📅 Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    versions_available = sorted(df_full["appVersion"].dropna().unique())
    ver_options = ["All versions"] + versions_available
    selected_versions = st.multiselect("📱 App Version", ver_options, default=["All versions"])

    sentiment_filter = st.multiselect("💬 Sentiment",
        ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

    search_term = st.text_input("🔍 Keyword Search", placeholder="e.g. delivery, refund…")

    st.divider()
    st.caption("Built with Streamlit + VADER + TF-IDF")

# ── Apply filters ─────────────────────────────────────────────────────────────
d0, d1 = (date_range[0], date_range[1]) if len(date_range) == 2 else (date_range[0], date_range[0])

use_versions = [] if "All versions" in selected_versions else selected_versions
df = filter_data(df_full, score_range, use_versions, (d0, d1))
df = df[df["sentiment"].isin(sentiment_filter)]
if search_term:
    df = df[df["content"].str.contains(search_term, case=False, na=False)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Amazon Shopping App — Review Intelligence Dashboard")
if df.empty:
    st.warning("No reviews match the current filters. Adjust the sidebar.")
    st.stop()

total_shown = len(df)
st.caption(f"Showing **{total_shown:,}** of {len(df_full):,} reviews · "
           f"Avg rating: **{df['score'].mean():.2f}** ⭐ · "
           f"Sentiment: {(df['sentiment']=='Positive').mean()*100:.0f}% positive")

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
avg_r   = df["score"].mean()
avg_s   = df["compound"].mean()
pct_pos = (df["sentiment"] == "Positive").mean() * 100
pct_neg = (df["sentiment"] == "Negative").mean() * 100
top_up  = df["thumbsUpCount"].max()

k1.metric("⭐ Avg Rating",     f"{avg_r:.2f} / 5")
k2.metric("💬 Avg Sentiment",  f"{avg_s:+.3f}", delta=f"{'Above' if avg_s > 0 else 'Below'} neutral")
k3.metric("😊 Positive",       f"{pct_pos:.1f}%")
k4.metric("😠 Negative",       f"{pct_neg:.1f}%")
k5.metric("👍 Max Upvotes",    f"{top_up:,}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_trends, tab_sentiment, tab_keywords, tab_complaints, tab_reviews = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "💬 Sentiment",
    "🔑 Keywords",
    "🚨 Complaints",
    "📝 Reviews",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = df["score"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Stars", "Count"]
        rating_counts["Color"] = rating_counts["Stars"].map(RATING_COLORS)
        rating_counts["Label"] = rating_counts["Stars"].astype(int).astype(str) + " ★"
        fig = px.bar(
            rating_counts, x="Label", y="Count",
            color="Label",
            color_discrete_map={r["Label"]: r["Color"] for _, r in rating_counts.iterrows()},
            text="Count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=330,
                          xaxis_title="Star Rating", yaxis_title="Reviews")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Positive vs Negative Breakdown</div>', unsafe_allow_html=True)
        pos_n = (df["score"] >= 4).sum()
        neu_n = (df["score"] == 3).sum()
        neg_n = (df["score"] <= 2).sum()
        fig_pie = go.Figure(go.Pie(
            labels=["Positive (4-5★)", "Neutral (3★)", "Negative (1-2★)"],
            values=[pos_n, neu_n, neg_n],
            marker_colors=["#2ca02c", "#ff7f0e", "#d62728"],
            hole=0.5,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(margin=dict(t=10, b=10), height=330, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    neg_pct = neg_n / total_shown * 100
    st.markdown(f"""
    <div class="insight-box">
    🔴 <b>{neg_pct:.1f}%</b> of filtered reviews are negative (1-2★). Amazon's global Play Store average
    is ~2.7★ — far below the 4.0★ benchmark for top shopping apps.
    </div>
    <div class="insight-box">
    🟡 The <b>3-star</b> tier ({neu_n:,} reviews) is underrepresented, suggesting users feel strongly
    either way — there's little middle ground.
    </div>
    <div class="insight-box">
    🟢 <b>Positive reviews</b> ({pos_n:,}) praise selection, pricing, and Prime convenience.
    These are leverage points to double down on.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Year-over-Year Summary</div>', unsafe_allow_html=True)
    yearly = (
        df.groupby("year")
        .agg(
            Reviews=("reviewId", "count"),
            Avg_Rating=("score", "mean"),
            Pct_1Star=("score", lambda x: (x == 1).mean() * 100),
            Pct_5Star=("score", lambda x: (x == 5).mean() * 100),
            Avg_Sentiment=("compound", "mean"),
        )
        .round(2)
        .reset_index()
    )
    yearly.columns = ["Year", "Reviews", "Avg Rating", "% 1-Star", "% 5-Star", "Avg Sentiment"]
    st.dataframe(yearly, use_container_width=True, hide_index=True,
                 column_config={
                     "Reviews":       st.column_config.NumberColumn(format="%d"),
                     "Avg Rating":    st.column_config.ProgressColumn(min_value=1, max_value=5, format="%.2f"),
                     "% 1-Star":      st.column_config.NumberColumn(format="%.1f%%"),
                     "% 5-Star":      st.column_config.NumberColumn(format="%.1f%%"),
                     "Avg Sentiment": st.column_config.NumberColumn(format="%.3f"),
                 })

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    monthly = (
        df.groupby("month")
        .agg(
            reviews=("reviewId", "count"),
            avg_rating=("score", "mean"),
            avg_sentiment=("compound", "mean"),
            pct_negative=("rating_label", lambda x: (x == "Negative").mean() * 100),
        )
        .reset_index().sort_values("month")
    )
    monthly["rolling_rating"] = monthly["avg_rating"].rolling(3, center=True, min_periods=1).mean()
    monthly["rolling_sentiment"] = monthly["avg_sentiment"].rolling(3, center=True, min_periods=1).mean()

    st.markdown('<div class="section-header">Monthly Review Volume</div>', unsafe_allow_html=True)
    fig_vol = px.area(monthly, x="month", y="reviews",
                      labels={"month": "Month", "reviews": "Reviews"},
                      color_discrete_sequence=["#1f77b4"])
    fig_vol.update_layout(margin=dict(t=10, b=10), height=260, xaxis_tickangle=-30)
    st.plotly_chart(fig_vol, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown('<div class="section-header">Avg Rating Over Time (3-month rolling)</div>', unsafe_allow_html=True)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_rating"],
                                   mode="lines", name="Monthly avg",
                                   line=dict(color="lightgray", width=1), opacity=0.6))
        fig_r.add_trace(go.Scatter(x=monthly["month"], y=monthly["rolling_rating"],
                                   mode="lines", name="3-month rolling",
                                   line=dict(color="#d62728", width=2.5)))
        fig_r.add_hline(y=df_full["score"].mean(), line_dash="dash", line_color="steelblue",
                        annotation_text=f"Overall avg ({df_full['score'].mean():.2f}★)")
        fig_r.update_layout(yaxis=dict(range=[1, 5], title="Avg Rating"),
                             margin=dict(t=10, b=10), height=300,
                             xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_r, use_container_width=True)

    with col_t2:
        st.markdown('<div class="section-header">Avg Sentiment Over Time (3-month rolling)</div>', unsafe_allow_html=True)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_sentiment"],
                                   mode="lines", name="Monthly",
                                   line=dict(color="lightgray", width=1), opacity=0.6))
        fig_s.add_trace(go.Scatter(x=monthly["month"], y=monthly["rolling_sentiment"],
                                   mode="lines", name="3-month rolling",
                                   line=dict(color="#2ca02c", width=2.5)))
        fig_s.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Neutral")
        fig_s.update_layout(yaxis=dict(range=[-1, 1], title="Compound Score"),
                             margin=dict(t=10, b=10), height=300,
                             xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-header">% Negative Reviews Over Time</div>', unsafe_allow_html=True)
    fig_neg = px.line(monthly, x="month", y="pct_negative",
                      labels={"month": "Month", "pct_negative": "% Negative Reviews"},
                      color_discrete_sequence=["#d62728"])
    fig_neg.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% threshold")
    fig_neg.update_layout(margin=dict(t=10, b=10), height=250, xaxis_tickangle=-30)
    st.plotly_chart(fig_neg, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown('<div class="section-header">Sentiment Distribution (VADER)</div>', unsafe_allow_html=True)
        sc = df["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment", "Count"]
        fig_sd = px.pie(sc, names="Sentiment", values="Count",
                        color="Sentiment", color_discrete_map=PALETTE, hole=0.45)
        fig_sd.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_sd, use_container_width=True)

    with col_s2:
        st.markdown('<div class="section-header">Avg Sentiment Score by Star Rating</div>', unsafe_allow_html=True)
        avg_cpd = df.groupby("score")["compound"].mean().reset_index()
        avg_cpd.columns = ["Stars", "Avg Sentiment"]
        avg_cpd["Color"] = avg_cpd["Stars"].map(RATING_COLORS)
        fig_cs = px.bar(avg_cpd, x="Stars", y="Avg Sentiment",
                        color="Stars",
                        color_discrete_map={s: RATING_COLORS[s] for s in RATING_COLORS},
                        text=avg_cpd["Avg Sentiment"].round(3))
        fig_cs.update_traces(textposition="outside")
        fig_cs.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                              yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown('<div class="section-header">Rating vs Sentiment Agreement Matrix</div>', unsafe_allow_html=True)
    agree = pd.crosstab(df["rating_label"], df["sentiment"], normalize="index") * 100
    agree = agree.round(1)
    fig_hm = px.imshow(
        agree, text_auto=".1f", color_continuous_scale="RdYlGn",
        aspect="auto", zmin=0, zmax=100,
        labels=dict(x="VADER Sentiment", y="Star Rating Label", color="%"),
    )
    fig_hm.update_layout(margin=dict(t=10, b=10), height=280)
    st.plotly_chart(fig_hm, use_container_width=True)

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.markdown('<div class="section-header">High Rating but Negative Sentiment</div>', unsafe_allow_html=True)
        mixed_neg = df[(df["rating_label"] == "Positive") & (df["sentiment"] == "Negative")].nsmallest(5, "compound")
        for _, row in mixed_neg.iterrows():
            st.markdown(f"⭐{int(row['score'])} &nbsp; `compound: {row['compound']:.3f}`", unsafe_allow_html=True)
            st.caption(row["content"][:200])

    with col_s4:
        st.markdown('<div class="section-header">Low Rating but Positive Sentiment</div>', unsafe_allow_html=True)
        mixed_pos = df[(df["rating_label"] == "Negative") & (df["sentiment"] == "Positive")].nlargest(5, "compound")
        for _, row in mixed_pos.iterrows():
            st.markdown(f"⭐{int(row['score'])} &nbsp; `compound: {row['compound']:.3f}`", unsafe_allow_html=True)
            st.caption(row["content"][:200])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_keywords:
    pos_texts = df[df["rating_label"] == "Positive"]["content"]
    neg_texts = df[df["rating_label"] == "Negative"]["content"]

    with st.spinner("Extracting keywords with TF-IDF…"):
        top_pos = get_top_keywords(pos_texts, n=20)
        top_neg = get_top_keywords(neg_texts, n=20)

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown('<div class="section-header">Top Keywords — Positive Reviews (4-5★)</div>', unsafe_allow_html=True)
        fig_kp = px.bar(top_pos[::-1].reset_index(drop=True), x="score", y="keyword",
                        orientation="h", color="score",
                        color_continuous_scale="Greens", text=top_pos["score"].round(1))
        fig_kp.update_traces(textposition="outside")
        fig_kp.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10, b=10), height=480,
                              xaxis_title="TF-IDF Score", yaxis_title="")
        st.plotly_chart(fig_kp, use_container_width=True)

    with col_k2:
        st.markdown('<div class="section-header">Top Keywords — Negative Reviews (1-2★)</div>', unsafe_allow_html=True)
        fig_kn = px.bar(top_neg[::-1].reset_index(drop=True), x="score", y="keyword",
                        orientation="h", color="score",
                        color_continuous_scale="Reds", text=top_neg["score"].round(1))
        fig_kn.update_traces(textposition="outside")
        fig_kn.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10, b=10), height=480,
                              xaxis_title="TF-IDF Score", yaxis_title="")
        st.plotly_chart(fig_kn, use_container_width=True)

    st.markdown('<div class="section-header">Word Clouds</div>', unsafe_allow_html=True)
    wc_cols = st.columns(2)
    for col, texts, cmap, title in [
        (wc_cols[0], pos_texts, "Greens", "Positive Reviews"),
        (wc_cols[1], neg_texts, "Reds",   "Negative Reviews"),
    ]:
        combined = " ".join(texts.tolist())
        if len(combined.strip()) > 20:
            wc = WordCloud(width=700, height=380, background_color="white",
                           colormap=cmap, stopwords=EXTRA_STOPS,
                           max_words=100, collocations=True,
                           prefer_horizontal=0.8).generate(combined)
            fig_wc, ax = plt.subplots(figsize=(7, 3.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
            with col:
                st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPLAINTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_complaints:
    neg_df = df[df["score"] <= 2].copy()
    if neg_df.empty:
        st.info("No negative reviews in current filter.")
    else:
        with st.spinner("Classifying complaints…"):
            neg_df["categories"] = neg_df["content"].apply(classify_complaint)

        from collections import Counter
        all_cats = [c for cats in neg_df["categories"] for c in cats]
        cat_counts = pd.Series(Counter(all_cats)).sort_values(ascending=False)
        cat_pct    = (cat_counts / len(neg_df) * 100).round(1)

        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown('<div class="section-header">Complaint Categories (% of 1-2★ Reviews)</div>', unsafe_allow_html=True)
            df_cats = pd.DataFrame({"Category": cat_counts.index, "Count": cat_counts.values,
                                    "% of Neg Reviews": cat_pct.values})
            fig_cc = px.bar(df_cats[::-1].reset_index(drop=True), x="% of Neg Reviews", y="Category",
                            orientation="h", color="% of Neg Reviews",
                            color_continuous_scale="Reds_r",
                            text=df_cats["% of Neg Reviews"].round(1))
            fig_cc.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_cc.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10),
                                 height=max(380, len(cat_counts) * 32),
                                 xaxis_title="% of Negative Reviews", yaxis_title="")
            st.plotly_chart(fig_cc, use_container_width=True)

        with col_c2:
            st.markdown('<div class="section-header">Top Category Breakdown</div>', unsafe_allow_html=True)
            for cat, pct in cat_pct.head(6).items():
                st.metric(cat, f"{pct:.1f}%")

        # Positive theme analysis
        pos_df2 = df[df["score"] >= 4].copy()
        pos_df2["themes"] = pos_df2["content"].apply(classify_positive)
        all_themes = [t for ts in pos_df2["themes"] for t in ts]
        theme_counts = pd.Series(Counter(all_themes)).sort_values(ascending=False)
        theme_pct    = (theme_counts / len(pos_df2) * 100).round(1)

        st.markdown('<div class="section-header">What Users Love (% of 4-5★ Reviews)</div>', unsafe_allow_html=True)
        df_themes = pd.DataFrame({"Theme": theme_counts.index, "Count": theme_counts.values,
                                  "% of Pos Reviews": theme_pct.values})
        fig_th = px.bar(df_themes[::-1].reset_index(drop=True), x="% of Pos Reviews", y="Theme",
                        orientation="h", color="% of Pos Reviews",
                        color_continuous_scale="Greens",
                        text=df_themes["% of Pos Reviews"].round(1))
        fig_th.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_th.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=350,
                             xaxis_title="% of Positive Reviews", yaxis_title="")
        st.plotly_chart(fig_th, use_container_width=True)

        # Sample reviews per category
        st.markdown('<div class="section-header">Sample Reviews per Complaint</div>', unsafe_allow_html=True)
        selected_cat = st.selectbox("Choose a complaint category", cat_counts.index.tolist())
        if selected_cat:
            samples = neg_df[neg_df["categories"].apply(lambda c: selected_cat in c)].head(5)
            for _, row in samples.iterrows():
                stars = "⭐" * int(row["score"])
                badge = f'<span class="tag-neg">Negative</span>'
                st.markdown(f"{stars} {badge} &nbsp; 👍{row['thumbsUpCount']} &nbsp; `{str(row['at'])[:10]}`",
                            unsafe_allow_html=True)
                st.write(f"> {row['content'][:400]}{'…' if len(row['content']) > 400 else ''}")
                st.divider()

    # App version analysis
    st.markdown('<div class="section-header">App Version Performance (Top/Bottom 10)</div>', unsafe_allow_html=True)
    ver_stats = (
        df.dropna(subset=["appVersion"])
        .groupby("appVersion")
        .agg(
            Avg_Rating=("score", "mean"),
            Reviews=("reviewId", "count"),
            Pct_Negative=("rating_label", lambda x: (x == "Negative").mean() * 100),
            Avg_Sentiment=("compound", "mean"),
        )
        .query("Reviews >= 30")
        .sort_values("Avg_Rating", ascending=False)
        .reset_index()
    )
    ver_stats.columns = ["Version", "Avg Rating", "Reviews", "% Negative", "Avg Sentiment"]

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        top10 = ver_stats.head(10)
        fig_tv = px.bar(top10[::-1].reset_index(drop=True), x="Avg Rating", y="Version",
                        orientation="h", color="Avg Rating",
                        color_continuous_scale="Greens", range_color=[1, 5],
                        text=top10["Avg Rating"].round(2))
        fig_tv.update_traces(textposition="outside")
        fig_tv.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=360,
                             title="Best Rated Versions (≥30 reviews)")
        st.plotly_chart(fig_tv, use_container_width=True)

    with col_v2:
        bot10 = ver_stats.tail(10).sort_values("Avg Rating")
        fig_bv = px.bar(bot10.reset_index(drop=True), x="Avg Rating", y="Version",
                        orientation="h", color="Avg Rating",
                        color_continuous_scale="Reds_r", range_color=[1, 5],
                        text=bot10["Avg Rating"].round(2))
        fig_bv.update_traces(textposition="outside")
        fig_bv.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), height=360,
                             title="Worst Rated Versions (≥30 reviews)")
        st.plotly_chart(fig_bv, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REVIEWS BROWSER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reviews:
    # Top upvoted
    st.markdown('<div class="section-header">Most Upvoted Reviews</div>', unsafe_allow_html=True)
    top5 = (df[df["thumbsUpCount"] > 0]
            .sort_values("thumbsUpCount", ascending=False)
            .head(5)
            .reset_index(drop=True))

    for _, row in top5.iterrows():
        stars = "⭐" * int(row["score"])
        sent_class = {"Positive": "tag-pos", "Negative": "tag-neg", "Neutral": "tag-neu"}.get(row["sentiment"], "tag-neu")
        st.markdown(
            f'**{row["userName"]}** &nbsp; {stars} &nbsp; '
            f'<span class="{sent_class}">{row["sentiment"]}</span> &nbsp; '
            f'👍 **{row["thumbsUpCount"]:,}** &nbsp; `{str(row["at"])[:10]}`',
            unsafe_allow_html=True,
        )
        st.write(f"> {row['content'][:400]}{'…' if len(row['content']) > 400 else ''}")
        st.divider()

    # Full browse
    st.markdown('<div class="section-header">Browse All Reviews</div>', unsafe_allow_html=True)
    sort_col = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "Rating (high)", "Rating (low)", "Upvotes (high)"])
    sort_map = {
        "Date (newest)": ("at", False), "Date (oldest)": ("at", True),
        "Rating (high)": ("score", False), "Rating (low)": ("score", True),
        "Upvotes (high)": ("thumbsUpCount", False),
    }
    scol, sasc = sort_map[sort_col]
    df_sorted = df.sort_values(scol, ascending=sasc)

    page_size = 25
    total_pages = max(1, (len(df_sorted) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size

    display = ["userName", "score", "sentiment", "compound", "thumbsUpCount", "appVersion", "at", "content"]
    st.dataframe(
        df_sorted[display].iloc[start: start + page_size].reset_index(drop=True),
        use_container_width=True,
        height=550,
        column_config={
            "score":        st.column_config.NumberColumn("⭐ Stars", format="%d"),
            "compound":     st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "thumbsUpCount":st.column_config.NumberColumn("👍 Upvotes"),
            "at":           st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
            "content":      st.column_config.TextColumn("Review", width="large"),
            "appVersion":   st.column_config.TextColumn("Version"),
            "sentiment":    st.column_config.TextColumn("Sentiment"),
        },
    )
    st.caption(f"Page {page} of {total_pages:,} · showing {min(page_size, len(df_sorted)-start)} of {len(df_sorted):,} reviews")
