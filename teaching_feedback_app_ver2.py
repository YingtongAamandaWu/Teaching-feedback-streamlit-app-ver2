# -*- coding: utf-8 -*-
"""Teaching_feedback_app_ver2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19s8Bt2RSsYgVWDuTaer-5yICfOzu5sx7

# Install required packages
"""

# Import packages

import streamlit as st
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download necessary NLTK resources
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK resources
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except:
    pass  # Skip downloading if an error occurs

# Ensure punkt tokenizer is available before using it
def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        st.error("Error: NLTK punkt tokenizer not found. Please reload the app.")
        return []



def summarize_text_sumy(text, algorithm="LSA", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizers = {
        "LSA": LsaSummarizer(),
        "Luhn": LuhnSummarizer(),
        "TextRank": TextRankSummarizer(),
        "LexRank": LexRankSummarizer()
    }
    summarizer = summarizers.get(algorithm, LsaSummarizer())
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# K-Means Clustering
def cluster_sentences_kmeans(sentences, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters, vectorizer, kmeans

# Extract Keywords from Clusters
def extract_cluster_keywords(vectorizer, kmeans, num_keywords=3):
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_cluster = []
    for cluster_idx in range(kmeans.n_clusters):
        top_indices = kmeans.cluster_centers_[cluster_idx].argsort()[-num_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        keywords_per_cluster.append(", ".join(keywords))
    return keywords_per_cluster

# Keyword-Based Clustering
def cluster_sentences_keywords(sentences, keyword_dict):
    cluster_labels = [-1] * len(sentences)
    for cluster_id, keywords in keyword_dict.items():
        for i, sentence in enumerate(sentences):
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                cluster_labels[i] = cluster_id
    return cluster_labels

st.title("Teaching Feedback Analyzer (beta-2.0)")
st.subheader("Utilizing the power of Natural Language Processing (NLP), unsupervised clustering, and text summary to empower teaching improvements!")
st.write("developed by Amanda Wu (ytwu@stanford.edu)")
st.write(" ") # intent to add space

st.subheader("Step 1 Overview",divider=True)
st.write("*Preview:* Dear fellow instructors and teaching teams, my goal for this app is to help analyze a large volume of text responses from survey questions, especially for a large-size class, such as BIO80s series. Feel free to test it out on your course feedback. This app does not capture user information, meaning that I as the app developer don't have access to your input data or who uses the app.") 
st.write(" ") # intent to add space
st.write("*Instructions:* You can download the course evaluation Excel sheet from the Stanford Course Eval system, then copy & paste students' responses for one of the open-ended questions (e.g., 'What would you like to say about this course to a student who is considering taking it in the future?'). On the app, you can select one of the text summary algorithms to summarize the course feedback (this is equivalent to a text summary function you see in Amazon product reviews). Meanwhile, you can exclude certain words that are less informative (such as 'course', 'class') from the wordcloud image visualization. The output of this app includes text summary, sentiment analysis (it analyzes sentiments behind the text, with positive value meaning positive sentiments),  a wordcloud image, and a table that show different categoiries of feedback based on text clustering.")

st.subheader("Step 2 Enter teaching feedback below and define methods of analysis", divider=True)
input_text = st.text_area("Please paste text of students' feedback for one survey question here:", height=150)
summarization_algorithm = st.selectbox("Select a summarization algorithm (see more about different summarizartion methods here https://miso-belica.github.io/sumy/summarizators.html):", ["LSA", "Luhn", "TextRank", "LexRank"], index=0)
clustering_method = st.radio("Select Clustering Method:", ["A) Unsupervised K-Means clustering", "B) Entering user-defined keywords"])

num_clusters = st.slider("IF you select 'A) Unsupervised K-Means clustering', please select the number of topic clusters:", 2, 10, 3)
keyword_input = st.text_area("IF you select 'B) Entering user-defined keywords', please mannually enter keywords for each topic that you wish to analyze. This will define the main clusters of feedback (comma-separated, one line per topic):", height=150)

# Input for Exclusion Words
excluded_words_input = st.text_input(
    "Enter words to exclude from the WordCloud (comma-separated):", 
    placeholder="e.g., class, course",
    key="excluded_words"
)

# Process Excluded Words
user_excluded_words = [word.strip().lower() for word in excluded_words_input.split(",") if word.strip()]

# Process Keyword-Based Clustering Input
keyword_dict = {}
if clustering_method == "B) Entering user-defined keywords" and keyword_input.strip():
    keyword_lines = keyword_input.split("\n")
    for idx, line in enumerate(keyword_lines):
        keyword_dict[idx] = [kw.strip() for kw in line.split(",") if kw.strip()]

st.subheader("Step 3 Result summary", divider=True)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        sentences = safe_sent_tokenize(input_text)
        polarities = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
        avg_sentiment = sum(polarities) / len(polarities) if polarities else 0
        sentiment_result = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"

        st.markdown(f"**Average Sentiment:** {sentiment_result} ({avg_sentiment:.2f})")
        st.markdown(f"**Sentence Count:** {len(sentences)}")

        summarized_text = summarize_text_sumy(input_text, algorithm=summarization_algorithm)
        st.subheader("3.1 Results: Text summary")
        st.write(summarized_text)

        st.subheader("3.2 Results: Sentiment analysis")
        st.write("Please find a boxplot summarizing the general sentiments from all students' comments below. You can hover over the scattered dots to see the original comment from each student. Polarity>0 means positive responses; polarity<0 means negative responses.")
        fig = px.box({"Sentence": sentences, "Polarity": polarities}, y="Polarity", points="all", hover_data=["Sentence"])
        st.plotly_chart(fig, use_container_width=True)

        stop_words = set(nltk.corpus.stopwords.words('english')).union(user_excluded_words)
        wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stop_words).generate(input_text)
        st.subheader("3.3 Result: WordCloud image")
        st.write("The wordcloud image summarizes the most frequently used words in the feedback. If you find some meaningless words that you would like to exclude, please go back to the textbox to enter words to be excluded, and hit 'Analyze' button again to update the results. Note that excluding these words will only affect the results in this wordcloud image, not other parts of the results.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        if sentences:
            if clustering_method == "A) Unsupervised K-Means clustering":
                clusters, vectorizer, kmeans = cluster_sentences_kmeans(sentences, num_clusters)
                cluster_keywords = extract_cluster_keywords(vectorizer, kmeans)
            else:
                clusters = cluster_sentences_keywords(sentences, keyword_dict)
                cluster_keywords = ["/".join(keyword_dict.get(i, [])) for i in range(len(keyword_dict))]

            cluster_data = pd.DataFrame({"Sentence": sentences, "Cluster": clusters})
            cluster_groups = {cluster: [] for cluster in set(clusters)}
            
            for index, row in cluster_data.iterrows():
                cluster_groups[row["Cluster"]].append(row["Sentence"])

            max_sentences = max(len(cluster_groups[c]) for c in cluster_groups)
            formatted_clusters = {f"Topic {i+1}: {cluster_keywords[i]}": cluster_groups[i] + [""] * (max_sentences - len(cluster_groups[i])) for i in range(len(cluster_keywords))}

            st.subheader("3.4 Result: Topic clusters")
            st.write("Each column shows the main topic representing one category/type of feedback.")
            st.dataframe(pd.DataFrame(formatted_clusters))


st.write("*Final words:* Thank you for using this app. This is a working project, so please don't hesitate to email me (ytwu@stanford.edu), if you have any questions, feedback, suggestions to share :) ") 
