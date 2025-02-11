#!/bin/bash
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet omw-1.4 brown
python -m textblob.download_corpora
