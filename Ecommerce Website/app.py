
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load dataset
df = pd.read_csv("test_data.csv")

# Convert labels for visualization
df["Sentiment"] = df["label"].map({1: "😊 Positive", 0: "😡 Negative"})

# Initialize user review session state
if "user_reviews" not in st.session_state:
    st.session_state["user_reviews"] = pd.DataFrame(columns=["Review", "Sentiment", "Confidence Score"])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Prediction function
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    vectorized = vectorizer.transform([cleaned_review])
    probs = model.predict_proba(vectorized)[0]
    sentiment = np.argmax(probs)
    confidence = probs[sentiment]
    return sentiment, confidence, probs

# UI Title
st.title("🛍️ E-Commerce Customer Review Sentiment Analysis")

# Text Input
review_input = st.text_area("📢 Enter a product review for analysis:")

if st.button("🔍 Analyze Sentiment"):
    if review_input:
        sentiment, confidence, probs = predict_sentiment(review_input)
        sentiment_text = "😊 Positive" if sentiment == 1 else "😡 Negative"

        # Add to session state
        new_review = pd.DataFrame([[review_input, sentiment_text, f"{confidence:.2f}"]],
                                  columns=["Review", "Sentiment", "Confidence Score"])
        st.session_state["user_reviews"] = pd.concat(
            [new_review, st.session_state["user_reviews"]],
            ignore_index=True
        )

        # Display Results
        st.write(f"**📝 Predicted Sentiment:** {sentiment_text}")
        st.write(f"**🔢 Confidence Score:** {confidence:.2f}")

        # Bar chart with actual prediction probabilities
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["😡 Negative", "😊 Positive"], probs, color=["red", "green"])
        ax.set_ylabel("Confidence Score")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# Show Submitted Reviews
if not st.session_state["user_reviews"].empty:
    st.subheader("📋 User Submitted Reviews")
    st.dataframe(st.session_state["user_reviews"])

# Combined dataset
df_combined = pd.concat([df, st.session_state["user_reviews"]], ignore_index=True)

# Sentiment Distribution
st.subheader("📊 Sentiment Distribution (Including User Inputs)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Sentiment", data=df_combined, palette={"😡 Negative": "red", "😊 Positive": "green"}, ax=ax)
ax.set_title("Sentiment Distribution")
st.pyplot(fig)

# Generate Word Clouds
pos_reviews = " ".join(df_combined[df_combined["Sentiment"] == "😊 Positive"]["Review"].astype(str).fillna(""))
neg_reviews = " ".join(df_combined[df_combined["Sentiment"] == "😡 Negative"]["Review"].astype(str).fillna(""))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Positive WordCloud
if pos_reviews.strip():
    pos_wc = WordCloud(width=400, height=300, background_color="white", max_words=100).generate(pos_reviews)
    ax[0].imshow(pos_wc, interpolation="bilinear")
    ax[0].axis("off")
    ax[0].set_title("😊 Positive Reviews")
else:
    ax[0].text(0.5, 0.5, "No Positive Reviews Yet", ha="center", va="center", fontsize=14)
    ax[0].axis("off")

# Negative WordCloud
if neg_reviews.strip():
    neg_wc = WordCloud(width=400, height=300, background_color="black", colormap="Reds", max_words=100).generate(neg_reviews)
    ax[1].imshow(neg_wc, interpolation="bilinear")
    ax[1].axis("off")
    ax[1].set_title("😡 Negative Reviews")
else:
    ax[1].text(0.5, 0.5, "No Negative Reviews Yet", ha="center", va="center", fontsize=14)
    ax[1].axis("off")

st.pyplot(fig)
