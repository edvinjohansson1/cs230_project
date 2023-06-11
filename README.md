# Predicting Informativeness of Product Reviews: A Deep Learning Approach
### Stanford University - CS230

This repository contains code for several neural network models to be run on Amazon review data to predict the helpfulness of each review.
We compare three supervised methods: a CNN-based, an LSTM-based, and a transformer-based method; as well as an unsupervised learning method focused on keyword extraction and matching.

To run these models yourself, you need to download a dataset from [Amazon customer review dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html). You also need to download [Stanford's GloVe 100d word embeddings](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt). Store these somewhere and update the respective paths in supervised/parameters.py.
