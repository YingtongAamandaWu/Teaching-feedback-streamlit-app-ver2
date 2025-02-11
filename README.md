# Teaching Feedback Analyzer (Beta 2.0)
![image](https://github.com/user-attachments/assets/761340fb-e556-4e47-9fb9-b4aaeeed1326)

## Overview
The **Teaching Feedback Analyzer** is a Streamlit-based web application designed to analyze student feedback from course evaluations using Natural Language Processing (NLP) techniques. This tool helps instructors and teaching teams extract insights from large volumes of textual responses, making it easier to understand student sentiments, identify key themes, and summarize feedback effectively.

## Features
- **Text Summarization**: Generates concise summaries using multiple NLP algorithms (LSA, Luhn, TextRank, LexRank).
- **Sentiment Analysis**: Evaluates the overall sentiment of student responses.
- **WordCloud Visualization**: Highlights the most frequently used words in feedback, with options to exclude less informative words.
- **Clustering Analysis**: Categorizes feedback into topics using:
  - **A) Unsupervised K-Means Clustering**
  - **B) User-Defined Keyword-Based Clustering**
- **Interactive Data Visualization**: Displays sentiment distribution using a boxplot with hoverable points.
- **Downloadable Report**: Allows users to download the analysis results as a PDF.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7) and install the required dependencies:
```sh
pip install -r requirements.txt
```

### Run the Application
```sh
streamlit run teaching_feedback_app_ver2.py
```

## Usage
1. **Paste** students' feedback responses into the text input field.
2. **Select** a text summarization algorithm.
3. **Choose** a clustering method:
   - Unsupervised K-Means (define number of clusters)
   - User-defined keyword-based clustering (enter keywords per topic)
4. **Specify** words to exclude from the WordCloud (optional).
5. **Analyze** the feedback by clicking the "Analyze" button.
6. **Review** the results, including:
   - Summary of feedback
   - Sentiment analysis with boxplot visualization
   - WordCloud representation
   - Feedback clustering output
7. **Download** the report as a PDF using the provided button.

## Dependencies
This application uses the following Python libraries:
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `wordcloud`
- `textblob`
- `sumy`
- `sklearn`
- `nltk`
- `reportlab`

## Contributing
If you would like to contribute, feel free to fork this repository and submit a pull request. Any feedback, feature requests, or bug reports are welcome!

## Contact
Developed by **Amanda Wu**
- 📧 Email: ytwu@stanford.edu
- 🔗 [LinkedIn](https://www.linkedin.com/in/yingtong-amanda-wu-48939021b/)
- 🏗️ [GitHub](https://github.com/YingtongAamandaWu)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

