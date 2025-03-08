import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fcmeans import FCM
import nltk
from nltk.tokenize import sent_tokenize
import io

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

from textblob.download_corpora import download_all
download_all()

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

def cluster_sentences_fcm(sentences, num_clusters=3):
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    # Convert to dense matrix if necessary
    X_dense = X.toarray()
    # Perform Fuzzy C-means clustering
    fcm = FCM(n_clusters=num_clusters)
    fcm.fit(X_dense)
    # Get cluster labels (hardened for simplicity)
    fcm_labels = fcm.u.argmax(axis=1)
    return fcm_labels, vectorizer, fcm

def extract_cluster_keywords_from_data(sentences, clusters, num_keywords=3):
    cluster_keywords = {}
    for cluster_id in set(clusters):
        cluster_sentences = [sentence for sentence, cluster in zip(sentences, clusters) if cluster == cluster_id]
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(cluster_sentences)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_values = np.array(X.sum(axis=0)).ravel()
        top_indices = np.argsort(tfidf_values)[-num_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices if i < len(feature_names)]
        cluster_keywords[cluster_id] = ", ".join(keywords)
    return cluster_keywords

def cluster_sentences_keywords(sentences, keyword_dict):
    cluster_labels = [-1] * len(sentences)
    for cluster_id, keywords in keyword_dict.items():
        for i, sentence in enumerate(sentences):
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                cluster_labels[i] = cluster_id
    return cluster_labels


st.title("Teaching Feedback Analyzer (beta-2.1)")
st.subheader("Utilizing the power of Natural Language Processing (NLP), unsupervised clustering, and text summary to empower teaching improvements!")
st.write("developed by Amanda Wu (ytwu@stanford.edu)")

st.write(" ") # intent to add space

st.write("*App Disclaimer:* This app does not store any data or personal information, and there are no plans for commercialization. The app is provided solely for demonstration and educational purposes.	This analysis is provided for your reference. Users should independently verify the information before making any decisions based on it.")

st.write(" ") # intent to add space

st.subheader("Step 1 Overview and instructions",divider=True)
st.write("*Preview:* Dear fellow instructors and teaching teams, my goal for this app is to help analyze a large volume of text responses from survey questions, especially for a large-size class, such as BIO80s series. Feel free to test it out on your course feedback. This app does not capture user information, meaning that I as the app developer don't have access to your input data or who uses the app.") 
st.write(" ") # intent to add space
st.write("*Instructions:* You can download the course evaluation Excel sheet from the Stanford Course Eval system, then copy & paste students' responses for one of the open-ended questions (e.g., 'What would you like to say about this course to a student who is considering taking it in the future?'). On the app, you can select one of the text summary algorithms to summarize the course feedback (this is equivalent to a text summary function you see in Amazon product reviews). Meanwhile, you can exclude certain words that are less informative (such as 'course', 'class') from the wordcloud image visualization. The output of this app includes text summary, sentiment analysis (it analyzes sentiments behind the text, with positive value meaning positive sentiments),  a wordcloud image, and a table that show different categoiries of feedback based on text clustering.")


st.subheader("Step 2 Enter teaching feedback and define methods for analysis", divider=True)
input_text = st.text_area("Please paste text of students' feedback for one survey question here:", height=150)
summarization_algorithm = st.selectbox("Select a summarization algorithm (see more about different summarizartion methods here https://miso-belica.github.io/sumy/summarizators.html):", ["LSA", "Luhn", "TextRank", "LexRank"], index=0)
clustering_method = st.radio("Select Clustering Method:", ["A) Unsupervised clustering", "B) Entering user-defined keywords"])

num_clusters = st.slider("IF you select 'A) Unsupervised clustering', please select the number of topic clusters:", 2, 10, 3)
keyword_input = st.text_area("IF you select 'B) Entering user-defined keywords', please mannually enter keywords for each topic that you wish to analyze. This will define how we categorize the feedback, and any other feedback that do not match to your keywords will be categorized to one cluster. Please define your keyword below (comma-separated, one line per topic):", height=150)

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
st.write("Please be patient while the analysis is running. This can take a minute.")


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
        st.subheader("3.1 Result: Text summary")
        st.write(summarized_text)

        st.subheader("3.2 Result: Sentiment analysis")
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
            if clustering_method == "A) Unsupervised clustering":
                clusters, vectorizer, fcm = cluster_sentences_fcm(sentences, num_clusters)
            else:
                clusters = cluster_sentences_keywords(sentences, keyword_dict)

            cluster_data = pd.DataFrame({"Sentence": sentences, "Cluster": clusters})

            cluster_groups = {}
            for index, row in cluster_data.iterrows():
                cluster_id = row["Cluster"]
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(row["Sentence"])

            max_sentences = max(len(cluster_groups[c]) for c in cluster_groups)

            cluster_keywords = extract_cluster_keywords_from_data(sentences, clusters, num_keywords=3)

            formatted_clusters = {}
            for i, cluster_id in enumerate(cluster_groups.keys()):
                topic_name = f"Topic {i+1}: {cluster_keywords[cluster_id]}"
                formatted_clusters[topic_name] = cluster_groups[cluster_id] + [""] * (max_sentences - len(cluster_groups[cluster_id]))

            st.subheader("3.4 Result: Topic clusters")
            st.write("Each column shows the main topic representing one category/type of feedback.")
            st.dataframe(pd.DataFrame(formatted_clusters))


st.subheader("Final words")
st.write("If you would like to download this report, please go to the vertical three dots on the upper right corner --> click 'Print' to export a PDF.") 
st.write("Thank you for using this app. This is a working project, so please don't hesitate to email me (ytwu@stanford.edu), if you have any questions, feedback, suggestions to share :) ") 

