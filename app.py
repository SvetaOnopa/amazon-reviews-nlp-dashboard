import os
import re
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from utils.preprocessing import (
    load_data, filter_data, get_top_keywords,
    COMPLAINT_CATEGORIES, POSITIVE_THEMES, KEYWORD_TOPICS,
    classify_complaint, classify_positive,
    SENTIMENT_CACHE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
.section-header {
    font-size: 1.05rem; font-weight: 700;
    border-left: 4px solid #1f77b4; padding-left: 10px;
    margin: 18px 0 8px 0;
}
.insight-box {
    background: #f7f9fc; border-left: 4px solid #ff7f0e;
    border-radius: 4px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.92rem;
}
.tag-pos { background:#e6f4ea; color:#1a7c34; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
.tag-neg { background:#fce8e6; color:#c5221f; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
.tag-neu { background:#fff3e0; color:#b45309; border-radius:4px; padding:2px 7px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

PALETTE       = {"Positive": "#2ca02c", "Neutral": "#ff7f0e", "Negative": "#d62728"}
RATING_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#ffdd57", 4: "#98df8a", 5: "#2ca02c"}
EXTRA_STOPS   = STOPWORDS | {
    "amazon", "app", "use", "one", "get", "now", "just", "like",
    "would", "got", "even", "know", "really", "much", "time",
    "still", "also", "used", "make", "give", "ve", "don", "ll",
}

_using_transformer = os.path.exists(SENTIMENT_CACHE)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & scoring 87K reviews…")
def get_data():
    return load_data("amazon_reviews.csv")

df_full = get_data()

_max_date = df_full["at"].dt.date.max()
_min_date = df_full["at"].dt.date.min()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📦 Amazon Reviews")
    st.caption(f"{len(df_full):,} reviews · {_min_date.year}–{_max_date.year}")

    st.divider()

    # ── Star rating ───────────────────────────────────────────────────────────
    st.subheader("⭐ Star Rating")
    score_range = st.slider("", 1, 5, (1, 5), label_visibility="collapsed")

    # ── Date range ────────────────────────────────────────────────────────────
    st.subheader("📅 Date Range")
    date_option = st.radio(
        "",
        ["All time", "Last year", "Last 6 months", "Last 3 months", "Last 30 days", "Last 7 days", "Custom"],
        index=0,
        label_visibility="collapsed",
    )
    if date_option == "Custom":
        _range = st.date_input(
            "Pick range",
            value=(_min_date, _max_date),
            min_value=_min_date,
            max_value=_max_date,
            label_visibility="collapsed",
        )
        if isinstance(_range, (list, tuple)) and len(_range) == 2:
            d0, d1 = _range[0], _range[1]
        else:
            d0 = d1 = _range if not isinstance(_range, (list, tuple)) else _range[0]
    else:
        d1 = _max_date
        _offsets = {
            "All time":      _min_date,
            "Last year":     _max_date - timedelta(days=365),
            "Last 6 months": _max_date - timedelta(days=182),
            "Last 3 months": _max_date - timedelta(days=91),
            "Last 30 days":  _max_date - timedelta(days=30),
            "Last 7 days":   _max_date - timedelta(days=7),
        }
        d0 = _offsets[date_option]

    # ── App version ───────────────────────────────────────────────────────────
    st.subheader("📱 App Version")
    versions_available = sorted(df_full["appVersion"].dropna().unique())
    selected_versions = st.multiselect(
        "",
        versions_available,
        default=[],
        placeholder="All versions",
        label_visibility="collapsed",
    )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    st.subheader("💬 Sentiment")
    sentiment_filter = st.multiselect(
        "",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
        label_visibility="collapsed",
    )

    # ── Topic / keyword ───────────────────────────────────────────────────────
    st.subheader("🔍 Topics / Keywords")
    selected_topics = st.multiselect(
        "",
        list(KEYWORD_TOPICS.keys()),
        default=[],
        placeholder="All topics (no filter)",
        label_visibility="collapsed",
    )

    st.divider()

# ── Apply filters ─────────────────────────────────────────────────────────────
df = filter_data(df_full, score_range, selected_versions, (d0, d1))
df = df[df["sentiment"].isin(sentiment_filter)]

if selected_topics:
    combined_pat = "|".join(KEYWORD_TOPICS[t] for t in selected_topics)
    df = df[df["content"].str.contains(combined_pat, case=False, na=False, regex=True)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Amazon Shopping App — Review Intelligence")

model_badge = "🤖 DistilBERT" if _using_transformer else "⚠️ VADER"
if df.empty:
    st.warning("No reviews match the current filters — adjust the sidebar.")
    st.stop()

st.caption(
    f"Showing **{len(df):,}** of {len(df_full):,} reviews · "
    f"Avg rating: **{df['score'].mean():.2f}★** · "
    f"{(df['sentiment']=='Positive').mean()*100:.0f}% positive · "
    f"Sentiment model: {model_badge}"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("⭐ Avg Rating",    f"{df['score'].mean():.2f} / 5")
k2.metric("💬 Avg Sentiment", f"{df['compound'].mean():+.3f}",
          delta="above neutral" if df["compound"].mean() > 0 else "below neutral",
          delta_color="normal" if df["compound"].mean() > 0 else "inverse")
k3.metric("😊 Positive",      f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
k4.metric("😠 Negative",      f"{(df['sentiment']=='Negative').mean()*100:.1f}%")
k5.metric("👍 Max Upvotes",   f"{df['thumbsUpCount'].max():,}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_trends, tab_sentiment, tab_keywords, tab_complaints, tab_reviews = st.tabs([
    "📊 Overview", "📈 Trends", "💬 Sentiment", "🔑 Keywords", "🚨 Complaints", "📝 Reviews",
])

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        rc = df["score"].value_counts().sort_index().reset_index()
        rc.columns = ["Stars", "Count"]
        rc["Label"] = rc["Stars"].astype(int).astype(str) + " ★"
        fig = px.bar(rc, x="Label", y="Count",
                     color="Label",
                     color_discrete_map={r["Label"]: RATING_COLORS[r["Stars"]] for _, r in rc.iterrows()},
                     text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=330,
                          xaxis_title="Star Rating", yaxis_title="Reviews")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Positive vs Negative Split</div>', unsafe_allow_html=True)
        pos_n = int((df["score"] >= 4).sum())
        neu_n = int((df["score"] == 3).sum())
        neg_n = int((df["score"] <= 2).sum())
        fig_pie = go.Figure(go.Pie(
            labels=["Positive (4-5★)", "Neutral (3★)", "Negative (1-2★)"],
            values=[pos_n, neu_n, neg_n],
            marker_colors=["#2ca02c", "#ff7f0e", "#d62728"],
            hole=0.5, textinfo="label+percent",
        ))
        fig_pie.update_layout(margin=dict(t=10, b=10), height=330, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box">🔴 <b>{neg_n/len(df)*100:.1f}%</b> of reviews are negative (1-2★).
    Amazon's Play Store average sits ~2.7★ — far below the 4.0★ benchmark for top shopping apps.</div>
    <div class="insight-box">🟡 The 3-star tier ({neu_n:,} reviews) is underrepresented — users
    feel strongly either way with little middle ground.</div>
    <div class="insight-box">🟢 Positive reviews ({pos_n:,}) consistently praise selection,
    pricing, and Prime convenience — leverage points to protect.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Year-over-Year Summary</div>', unsafe_allow_html=True)
    yearly = (
        df.groupby("year")
        .agg(Reviews=("reviewId","count"), Avg_Rating=("score","mean"),
             Pct_1Star=("score", lambda x: (x==1).mean()*100),
             Pct_5Star=("score", lambda x: (x==5).mean()*100),
             Avg_Sentiment=("compound","mean"))
        .round(2).reset_index()
    )
    yearly.columns = ["Year","Reviews","Avg Rating","% 1-Star","% 5-Star","Avg Sentiment"]
    st.dataframe(yearly, use_container_width=True, hide_index=True,
                 column_config={
                     "Reviews":       st.column_config.NumberColumn(format="%d"),
                     "Avg Rating":    st.column_config.ProgressColumn(min_value=1, max_value=5, format="%.2f"),
                     "% 1-Star":      st.column_config.NumberColumn(format="%.1f%%"),
                     "% 5-Star":      st.column_config.NumberColumn(format="%.1f%%"),
                     "Avg Sentiment": st.column_config.NumberColumn(format="%.3f"),
                 })

# ═══════════════════════════════════════════════════════════════════════════════
# TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    monthly = (
        df.groupby("month")
        .agg(reviews=("reviewId","count"), avg_rating=("score","mean"),
             avg_sentiment=("compound","mean"),
             pct_negative=("rating_label", lambda x: (x=="Negative").mean()*100))
        .reset_index().sort_values("month")
    )
    monthly["roll_rating"]    = monthly["avg_rating"].rolling(3, center=True, min_periods=1).mean()
    monthly["roll_sentiment"] = monthly["avg_sentiment"].rolling(3, center=True, min_periods=1).mean()

    st.markdown('<div class="section-header">Monthly Review Volume</div>', unsafe_allow_html=True)
    fig_vol = px.area(monthly, x="month", y="reviews",
                      color_discrete_sequence=["#1f77b4"],
                      labels={"month":"Month","reviews":"Reviews"})
    fig_vol.update_layout(margin=dict(t=10,b=10), height=240, xaxis_tickangle=-30)
    st.plotly_chart(fig_vol, use_container_width=True)

    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown('<div class="section-header">Avg Rating — 3-Month Rolling</div>', unsafe_allow_html=True)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_rating"],
                                   mode="lines", name="Monthly", line=dict(color="lightgray",width=1), opacity=0.5))
        fig_r.add_trace(go.Scatter(x=monthly["month"], y=monthly["roll_rating"],
                                   mode="lines", name="3-mo rolling", line=dict(color="#d62728",width=2.5)))
        fig_r.add_hline(y=df_full["score"].mean(), line_dash="dash", line_color="steelblue",
                        annotation_text=f"Overall avg ({df_full['score'].mean():.2f}★)")
        fig_r.update_layout(yaxis=dict(range=[1,5],title="Avg Rating"),
                             margin=dict(t=10,b=10), height=280, xaxis_tickangle=-30,
                             legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_r, use_container_width=True)

    with ct2:
        st.markdown('<div class="section-header">Avg Sentiment — 3-Month Rolling</div>', unsafe_allow_html=True)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=monthly["month"], y=monthly["avg_sentiment"],
                                   mode="lines", name="Monthly", line=dict(color="lightgray",width=1), opacity=0.5))
        fig_s.add_trace(go.Scatter(x=monthly["month"], y=monthly["roll_sentiment"],
                                   mode="lines", name="3-mo rolling", line=dict(color="#2ca02c",width=2.5)))
        fig_s.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Neutral")
        fig_s.update_layout(yaxis=dict(range=[-1,1],title="Compound Score"),
                             margin=dict(t=10,b=10), height=280, xaxis_tickangle=-30,
                             legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-header">% Negative Reviews Over Time</div>', unsafe_allow_html=True)
    fig_neg = px.line(monthly, x="month", y="pct_negative",
                      color_discrete_sequence=["#d62728"],
                      labels={"month":"Month","pct_negative":"% Negative Reviews"})
    fig_neg.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig_neg.update_layout(margin=dict(t=10,b=10), height=240, xaxis_tickangle=-30)
    st.plotly_chart(fig_neg, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    cs1, cs2 = st.columns(2)

    with cs1:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        sc = df["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment","Count"]
        fig_sd = px.pie(sc, names="Sentiment", values="Count",
                        color="Sentiment", color_discrete_map=PALETTE, hole=0.45)
        fig_sd.update_layout(margin=dict(t=10,b=10), height=300)
        st.plotly_chart(fig_sd, use_container_width=True)

    with cs2:
        st.markdown('<div class="section-header">Avg Sentiment by Star Rating</div>', unsafe_allow_html=True)
        avg_cpd = df.groupby("score")["compound"].mean().reset_index()
        avg_cpd.columns = ["Stars","Avg Sentiment"]
        fig_cs = px.bar(avg_cpd, x="Stars", y="Avg Sentiment",
                        color="Stars", color_discrete_map={s: RATING_COLORS[s] for s in RATING_COLORS},
                        text=avg_cpd["Avg Sentiment"].round(3))
        fig_cs.update_traces(textposition="outside")
        fig_cs.update_layout(showlegend=False, margin=dict(t=10,b=10), height=300,
                              yaxis=dict(range=[-1,1]))
        st.plotly_chart(fig_cs, use_container_width=True)

    st.caption(
        "Sentiment is assigned by DistilBERT: **Positive** = model is ≥72% confident the text is positive · "
        "**Negative** = ≥72% confident negative · **Neutral** = model is uncertain (confidence <72%). "
        "The Avg Sentiment score ranges from −1 (most negative) to +1 (most positive)."
    )
    st.markdown('<div class="section-header">Rating vs Sentiment Agreement Matrix</div>', unsafe_allow_html=True)
    agree = pd.crosstab(df["rating_label"], df["sentiment"], normalize="index") * 100
    fig_hm = px.imshow(agree.round(1), text_auto=".1f", color_continuous_scale="RdYlGn",
                       aspect="auto", zmin=0, zmax=100,
                       labels=dict(x="Sentiment (model)", y="Star Rating Group", color="%"))
    fig_hm.update_layout(margin=dict(t=10,b=10), height=260)
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        "Each row shows the % of reviews in that star-rating group that received each sentiment label (rows sum to 100%). "
        "Ideally: high-star reviews → Positive, low-star reviews → Negative. "
        "Off-diagonal cells reveal disagreements — e.g. a 5★ review the model reads as Negative, or a 1★ review with genuinely positive text."
    )

    cs3, cs4 = st.columns(2)
    with cs3:
        st.markdown('<div class="section-header">⚠️ High Rating but Negative Sentiment</div>', unsafe_allow_html=True)
        mixed_neg = (df[(df["rating_label"]=="Positive") & (df["sentiment"]=="Negative")]
                     .nsmallest(5,"compound"))
        for _, row in mixed_neg.iterrows():
            st.markdown(f"⭐{int(row['score'])} &nbsp; `score: {row['compound']:.3f}`",
                        unsafe_allow_html=True)
            st.caption(row["content"][:250])

    with cs4:
        st.markdown('<div class="section-header">⚠️ Low Rating but Positive Sentiment</div>', unsafe_allow_html=True)
        mixed_pos = (df[(df["rating_label"]=="Negative") & (df["sentiment"]=="Positive")]
                     .nlargest(5,"compound"))
        for _, row in mixed_pos.iterrows():
            st.markdown(f"⭐{int(row['score'])} &nbsp; `score: {row['compound']:.3f}`",
                        unsafe_allow_html=True)
            st.caption(row["content"][:250])

# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_keywords:
    pos_texts = df[df["rating_label"]=="Positive"]["content"]
    neg_texts = df[df["rating_label"]=="Negative"]["content"]

    with st.spinner("Extracting TF-IDF keywords…"):
        top_pos = get_top_keywords(pos_texts, n=20)
        top_neg = get_top_keywords(neg_texts, n=20)

    ck1, ck2 = st.columns(2)
    with ck1:
        st.markdown('<div class="section-header">Top Keywords — Positive (4-5★)</div>', unsafe_allow_html=True)
        _kp_df = top_pos[::-1].reset_index(drop=True)
        fig_kp = px.bar(_kp_df, x="score", y="keyword",
                        orientation="h", color="score",
                        color_continuous_scale="Greens", text=_kp_df["score"].round(1))
        fig_kp.update_traces(textposition="outside")
        fig_kp.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10), height=480,
                              xaxis_title="TF-IDF Score", yaxis_title="")
        st.plotly_chart(fig_kp, use_container_width=True)

    with ck2:
        st.markdown('<div class="section-header">Top Keywords — Negative (1-2★)</div>', unsafe_allow_html=True)
        _kn_df = top_neg[::-1].reset_index(drop=True)
        fig_kn = px.bar(_kn_df, x="score", y="keyword",
                        orientation="h", color="score",
                        color_continuous_scale="Reds", text=_kn_df["score"].round(1))
        fig_kn.update_traces(textposition="outside")
        fig_kn.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=10,b=10), height=480,
                              xaxis_title="TF-IDF Score", yaxis_title="")
        st.plotly_chart(fig_kn, use_container_width=True)

    st.caption("**TF-IDF Score** — measures how characteristic a word is for this review group. It rewards words that appear frequently here but rarely in the opposite group. Higher score = more distinctive keyword. Darker color = higher score. Bars are sorted top-to-bottom from most to least distinctive.")

    st.markdown('<div class="section-header">Word Clouds</div>', unsafe_allow_html=True)
    wc_cols = st.columns(2)
    for col, texts, cmap, title in [
        (wc_cols[0], pos_texts, "Greens", "Positive Reviews (4-5★)"),
        (wc_cols[1], neg_texts, "Reds",   "Negative Reviews (1-2★)"),
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
# COMPLAINTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_complaints:
    from collections import Counter

    neg_df = df[df["score"] <= 2].copy()
    if neg_df.empty:
        st.info("No negative reviews in current filter.")
    else:
        with st.spinner("Classifying complaints…"):
            neg_df["categories"] = neg_df["content"].apply(classify_complaint)

        all_cats   = [c for cats in neg_df["categories"] for c in cats]
        cat_counts = pd.Series(Counter(all_cats)).sort_values(ascending=False)
        cat_pct    = (cat_counts / len(neg_df) * 100).round(1)

        cc1, cc2 = st.columns([2, 1])
        with cc1:
            st.markdown('<div class="section-header">Complaint Categories (% of 1-2★)</div>', unsafe_allow_html=True)
            df_cats = pd.DataFrame({"Category": cat_counts.index,
                                    "Count": cat_counts.values,
                                    "% Neg Reviews": cat_pct.values})
            _cc_df = df_cats[::-1].reset_index(drop=True)
            fig_cc = px.bar(_cc_df,
                            x="% Neg Reviews", y="Category", orientation="h",
                            color="% Neg Reviews", color_continuous_scale="Reds",
                            text=_cc_df["% Neg Reviews"].round(1))
            fig_cc.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_cc.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10),
                                 height=max(380, len(cat_counts)*32),
                                 xaxis_title="% of Negative Reviews", yaxis_title="")
            st.plotly_chart(fig_cc, use_container_width=True)
            st.caption("% = share of 1-2★ reviews that mention this topic. One review can match multiple categories, so totals may exceed 100%. Darker red = more prevalent complaint. Sorted top-to-bottom from most to least common.")

        with cc2:
            st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
            for cat, pct in cat_pct.head(6).items():
                st.metric(cat, f"{pct:.1f}%")

        if "is_sarcastic" in neg_df.columns:
            sarcasm_df = neg_df[neg_df["is_sarcastic"]].sort_values("thumbsUpCount", ascending=False)
            if not sarcasm_df.empty:
                st.markdown('<div class="section-header">🙄 Frustrated Sarcasm Detected</div>', unsafe_allow_html=True)
                pct_sarc = len(sarcasm_df) / len(neg_df) * 100
                st.caption(
                    f"**{len(sarcasm_df):,}** reviews ({pct_sarc:.1f}% of 1-2★) flagged as likely sarcastic "
                    f"(laughing emojis in negative context, suspiciously high sentiment score on low rating, "
                    f"or sarcastic phrases). Sentiment auto-corrected to Negative."
                )
                for _, row in sarcasm_df.head(4).iterrows():
                    stars = "⭐" * int(row["score"])
                    st.markdown(
                        f"{stars} &nbsp; 👍{row['thumbsUpCount']} &nbsp; `{str(row['at'])[:10]}`",
                        unsafe_allow_html=True,
                    )
                    st.write(f"> {row['content'][:400]}{'…' if len(row['content']) > 400 else ''}")
                    st.divider()

    # Positive themes
    pos_df2 = df[df["score"] >= 4].copy()
    pos_df2["themes"] = pos_df2["content"].apply(classify_positive)
    all_themes   = [t for ts in pos_df2["themes"] for t in ts]
    theme_counts = pd.Series(Counter(all_themes)).sort_values(ascending=False)
    theme_pct    = (theme_counts / len(pos_df2) * 100).round(1)

    st.markdown('<div class="section-header">What Users Love (% of 4-5★)</div>', unsafe_allow_html=True)
    df_th = pd.DataFrame({"Theme": theme_counts.index,
                          "Count": theme_counts.values,
                          "% Pos Reviews": theme_pct.values})
    _th_df = df_th[::-1].reset_index(drop=True)
    fig_th = px.bar(_th_df,
                    x="% Pos Reviews", y="Theme", orientation="h",
                    color="% Pos Reviews", color_continuous_scale="Greens",
                    text=_th_df["% Pos Reviews"].round(1))
    fig_th.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_th.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10), height=320,
                         xaxis_title="% of Positive Reviews", yaxis_title="")
    st.plotly_chart(fig_th, use_container_width=True)

    # Sample reviews per category
    if not neg_df.empty:
        st.markdown('<div class="section-header">Sample Reviews per Complaint</div>', unsafe_allow_html=True)
        sel_cat = st.selectbox("Choose a category", cat_counts.index.tolist())
        if sel_cat:
            samples = neg_df[neg_df["categories"].apply(lambda c: sel_cat in c)].head(5)
            for _, row in samples.iterrows():
                stars = "⭐" * int(row["score"])
                st.markdown(f"{stars} &nbsp; 👍{row['thumbsUpCount']} &nbsp; `{str(row['at'])[:10]}`",
                            unsafe_allow_html=True)
                st.write(f"> {row['content'][:400]}{'…' if len(row['content'])>400 else ''}")
                st.divider()

    # Version performance
    st.markdown('<div class="section-header">App Version Performance</div>', unsafe_allow_html=True)
    ver_stats = (
        df.dropna(subset=["appVersion"])
        .groupby("appVersion")
        .agg(Avg_Rating=("score","mean"), Reviews=("reviewId","count"),
             Pct_Negative=("rating_label", lambda x: (x=="Negative").mean()*100),
             Avg_Sentiment=("compound","mean"))
        .query("Reviews >= 30")
        .sort_values("Avg_Rating", ascending=False)
        .reset_index()
    )
    ver_stats.columns = ["Version","Avg Rating","Reviews","% Negative","Avg Sentiment"]

    cv1, cv2 = st.columns(2)
    with cv1:
        top10 = ver_stats.head(10)
        _tv_df = top10[::-1].reset_index(drop=True)
        fig_tv = px.bar(_tv_df, x="Avg Rating", y="Version",
                        orientation="h", color="Avg Rating",
                        color_continuous_scale="Greens", range_color=[1,5],
                        text=_tv_df["Avg Rating"].round(2))
        fig_tv.update_traces(textposition="outside")
        fig_tv.update_layout(coloraxis_showscale=False, margin=dict(t=30,b=10), height=360,
                             title="Best Rated Versions (≥30 reviews)")
        st.plotly_chart(fig_tv, use_container_width=True)

    with cv2:
        bot10 = ver_stats.tail(10).sort_values("Avg Rating")
        fig_bv = px.bar(bot10.reset_index(drop=True), x="Avg Rating", y="Version",
                        orientation="h", color="Avg Rating",
                        color_continuous_scale="Reds_r", range_color=[1,5],
                        text=bot10["Avg Rating"].round(2))
        fig_bv.update_traces(textposition="outside")
        fig_bv.update_layout(coloraxis_showscale=False, margin=dict(t=30,b=10), height=360,
                             title="Worst Rated Versions (≥30 reviews)")
        st.plotly_chart(fig_bv, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REVIEWS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reviews:
    st.markdown('<div class="section-header">Most Upvoted Reviews</div>', unsafe_allow_html=True)
    top5 = (df[df["thumbsUpCount"] > 0]
            .sort_values("thumbsUpCount", ascending=False)
            .head(5).reset_index(drop=True))

    for _, row in top5.iterrows():
        stars      = "⭐" * int(row["score"])
        sent_class = {"Positive":"tag-pos","Negative":"tag-neg","Neutral":"tag-neu"}.get(row["sentiment"],"tag-neu")
        st.markdown(
            f'**{row["userName"]}** &nbsp; {stars} &nbsp; '
            f'<span class="{sent_class}">{row["sentiment"]}</span> &nbsp; '
            f'👍 **{row["thumbsUpCount"]:,}** &nbsp; `{str(row["at"])[:10]}`',
            unsafe_allow_html=True,
        )
        st.write(f"> {row['content'][:400]}{'…' if len(row['content'])>400 else ''}")
        st.divider()

    st.markdown('<div class="section-header">Browse All Reviews</div>', unsafe_allow_html=True)
    sort_col = st.selectbox("Sort by", [
        "Date (newest)", "Date (oldest)",
        "Rating (high)", "Rating (low)",
        "Upvotes (high)", "Sentiment (most positive)", "Sentiment (most negative)",
    ])
    sort_map = {
        "Date (newest)":           ("at",            False),
        "Date (oldest)":           ("at",            True),
        "Rating (high)":           ("score",         False),
        "Rating (low)":            ("score",         True),
        "Upvotes (high)":          ("thumbsUpCount", False),
        "Sentiment (most positive)":("compound",     False),
        "Sentiment (most negative)":("compound",     True),
    }
    scol, sasc = sort_map[sort_col]
    df_sorted  = df.sort_values(scol, ascending=sasc)

    page_size   = 25
    total_pages = max(1, (len(df_sorted) - 1) // page_size + 1)
    page        = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start       = (page - 1) * page_size

    display_cols = ["userName","score","sentiment","compound","thumbsUpCount","appVersion","at","content"]
    st.dataframe(
        df_sorted[display_cols].iloc[start: start+page_size].reset_index(drop=True),
        use_container_width=True, height=550,
        column_config={
            "score":         st.column_config.NumberColumn("⭐ Stars",   format="%d"),
            "compound":      st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "thumbsUpCount": st.column_config.NumberColumn("👍 Upvotes"),
            "at":            st.column_config.DatetimeColumn("Date",    format="YYYY-MM-DD"),
            "content":       st.column_config.TextColumn("Review",      width="large"),
            "appVersion":    st.column_config.TextColumn("Version"),
            "sentiment":     st.column_config.TextColumn("Sentiment"),
        },
    )
    st.caption(f"Page {page} of {total_pages:,} · {len(df_sorted):,} reviews")
