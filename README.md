# Wine Recommender

### Introduction

In this repository, we (i) explore an approach to generating an embedding for each wine review that combines some of the best-practice approaches highlighted in existing literature, and (ii) use these wine embeddings to build a simple wine recommendation engine. This repository contains the following files:

- creating_wine_review_embeddings.ipynb: this is the notebook file containing the analysis creating the wine embeddings, and building the wine recommendation engine
- descriptor_mapping.csv: the full mapping of preprocessed wine descriptors to level 1, level 2 and level 3 terms.
- wine_word2vec_model.bin, wine_word2vec_model.bin.trainables.syn1neg.npy, wine_word2vec_model.bin.wv.vectors.npy: trained instances of the word2vec model trained on the wine review corpus

The data obtained from the web scraping exercise was too large to be added to this repository. However, the scraper used to mine the data from www.winemag.com is available in this GitHub respository: https://github.com/RoaldSchuring/studying_grape_styles

### Technologies

- Python
- Jupyter Notebook
- The necessary Python package versions needed to run the various files in this repository have been listed out in the accompanying requirements.txt file

### Project Description

One of the cornerstones of every chapter of the Robosomm series has been to extract descriptors from professional wine reviews, and to convert these into quantitative features. In doing so, we want to put ourselves in the shoes of a blind taster and extract only those descriptors that could be derived without knowing what the wine actually is.

In this notebook, we will combine a couple of best-practice approaches highlighted in the existing literature on this subject and create an embedding for each wine review. We will use this to build a simple wine recommender.

### Getting Started

1. Clone this repo

2. Run the web scraper available in the other repository outlined above to get a full and fresh set of wine reviews

3. Swap in the location of the csv files with the scraped wine reviews for the hard-coded location in section 1 of creating_wine_review_embeddings.ipynb

4. Run creating_wine_review_embeddings.ipynb as you please to replicate the analysis.

### Authors

Roald Schuring
