import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Review Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
)

# ── Load DistilBERT (pre-trained, no fine-tuning needed) ─────────────────────
@st.cache_resource(show_spinner=False)
def load_distilbert():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )

# ── Train TF-IDF + Logistic Regression on IMDB subset ────────────────────────
@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    from datasets import load_dataset
    ds = load_dataset("imdb", split="train[:2000]")
    texts  = ds["text"]
    labels = ds["label"]          # 0 = negative, 1 = positive

    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english")
    X   = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, labels)
    return vec, clf

# ── Example reviews ───────────────────────────────────────────────────────────
EXAMPLES = {
    "⭐⭐⭐⭐⭐  Glowing review": (
        "An absolute masterpiece. The performances were extraordinary, the cinematography "
        "breathtaking, and the story deeply moving. One of the best films I've seen in years. "
        "I left the theater in tears — the good kind."
    ),
    "💩  Scathing review": (
        "Complete waste of time and money. The plot made no sense, the acting was wooden, "
        "and the CGI looked like it was made in 2003. I fell asleep twice. "
        "Do yourself a favor and skip this one entirely."
    ),
    "🤷  Mixed review": (
        "It had its moments — some genuinely funny scenes and a strong lead performance — "
        "but the pacing dragged badly in the second act and the ending felt rushed. "
        "Worth a watch, but don't go in with high expectations."
    ),
    "😏  Sarcastic review": (
        "Oh sure, another superhero sequel with a paper-thin villain and a plot twist "
        "telegraphed in the first five minutes. 'Groundbreaking.' The audience applauded "
        "at the end — I checked my watch."
    ),
    "✍️  Write your own": "",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Classifier")
st.markdown(
    "Paste any movie review and see how two NLP models classify its sentiment — "
    "a fine-tuned **DistilBERT** transformer and a classical **TF-IDF + Logistic Regression** baseline."
)
st.markdown("---")

# ── Load models ───────────────────────────────────────────────────────────────
col_l1, col_l2 = st.columns(2)
with col_l1:
    with st.spinner("Loading DistilBERT (~30 s on first launch)…"):
        bert_pipe = load_distilbert()
    st.caption("✅ DistilBERT ready")
with col_l2:
    with st.spinner("Training TF-IDF model…"):
        vec, clf = load_tfidf_model()
    st.caption("✅ TF-IDF model ready")

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📝 Enter a Review")

example_choice = st.selectbox("Try an example or write your own:", list(EXAMPLES.keys()))
default_text   = EXAMPLES[example_choice]

review_text = st.text_area(
    "Review text:",
    value=default_text,
    height=160,
    placeholder="Paste or type any movie review here…",
)

predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        st.markdown("---")
        st.subheader("📊 Results")

        # DistilBERT
        bert_result = bert_pipe(review_text[:512])[0]
        bert_label  = bert_result["label"]       # "POSITIVE" or "NEGATIVE"
        bert_conf   = bert_result["score"]

        # TF-IDF + LR
        X_input    = vec.transform([review_text])
        tfidf_pred = clf.predict(X_input)[0]
        tfidf_proba = clf.predict_proba(X_input)[0]
        tfidf_label = "POSITIVE" if tfidf_pred == 1 else "NEGATIVE"
        tfidf_conf  = float(tfidf_proba[tfidf_pred])

        def badge(label):
            return "🟢 POSITIVE" if label == "POSITIVE" else "🔴 NEGATIVE"

        col_b, col_t = st.columns(2)

        with col_b:
            st.markdown("### 🤖 DistilBERT")
            st.markdown(f"## {badge(bert_label)}")
            st.metric("Confidence", f"{bert_conf*100:.1f}%")
            st.progress(bert_conf)
            st.caption("Fine-tuned transformer · ~91% IMDB accuracy · 0.97 ROC-AUC")

        with col_t:
            st.markdown("### 📈 TF-IDF + Logistic Regression")
            st.markdown(f"## {badge(tfidf_label)}")
            st.metric("Confidence", f"{tfidf_conf*100:.1f}%")
            st.progress(tfidf_conf)
            st.caption("Bigram TF-IDF · trained on 2,000 IMDB reviews")

        # Agreement callout
        if bert_label == tfidf_label:
            st.success(f"✅ Both models agree: **{bert_label}**")
        else:
            st.warning(
                f"⚠️ Models disagree — DistilBERT says **{bert_label}**, "
                f"TF-IDF says **{tfidf_label}**. "
                "This often happens with sarcasm or nuanced language the bag-of-words model misses."
            )

        # Feature explainer
        with st.expander("🔍 What words drove the TF-IDF prediction?"):
            feature_names = np.array(vec.get_feature_names_out())
            coef          = clf.coef_[0]
            input_vec     = X_input.toarray()[0]
            active_idx    = np.where(input_vec > 0)[0]

            if len(active_idx) > 0:
                scores     = input_vec[active_idx] * coef[active_idx]
                sorted_idx = np.argsort(scores)

                pos_words = [
                    (feature_names[active_idx[i]], scores[i])
                    for i in sorted_idx[-5:][::-1] if scores[i] > 0
                ]
                neg_words = [
                    (feature_names[active_idx[i]], scores[i])
                    for i in sorted_idx[:5] if scores[i] < 0
                ]

                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Top positive signals:**")
                    for word, score in pos_words:
                        st.markdown(f"- `{word}` (+{score:.3f})")
                with col_neg:
                    st.markdown("**Top negative signals:**")
                    for word, score in neg_words:
                        st.markdown(f"- `{word}` ({score:.3f})")
            else:
                st.info("No recognisable TF-IDF features found in this text.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DistilBERT: `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face) · "
    "TF-IDF + Logistic Regression trained on IMDB reviews · "
    "[GitHub repo](https://github.com/robertciceroson/NLP-Movie-Review-Sentiment-Classifier)"
)
