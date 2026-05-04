#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
mkdir -p nltk_data
python -m nltk.downloader -d ./nltk_data movie_reviews stopwords punkt punkt_tab
