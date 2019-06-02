import numpy as np
import csv

import re
import nltk
from nltk.stem import SnowballStemmer
import string

def read_csv(filename):
  with open(filename, 'r') as f:
      it = csv.reader(f, delimiter = ',', quotechar = '"')
      data = [data for data in it]
  return data[1:] # Skip the first line

def load_data():
  train_data = read_csv("dataset/train.csv")
  test_data = [],
  #test_data = read_csv("dataset/test.csv")
  return train_data, None

"""
punctuation_table = str.maketrans('', '', string.punctuation)
# https://www.kaggle.com/currie32/the-importance-of-cleaning-text
# NLTK seems to have too many, esp the negation ones are essential.
# TODO: Some words may be critical to understanding the sentence -- may remove a few when looking at bad estimations.
stop_words = [
    "a", "i",
    "an", "as", "by", "at", "if", "in", "is", "of", "on", "so", "to",
    "the", "and", "but", "for",
    "this", "that", "these", "those", "then", "than",
    "what", "which", "while", "who",
    "because", "through", "during", "just", "about",
]
stemmer = SnowballStemmer('english')

def clean_text(text):
  text = text.lower()
  # https://www.kaggle.com/currie32/the-importance-of-cleaning-text
  text = re.sub(r"it's", "it is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"here's", "here is", text)
  text = re.sub(r"there's", "there is", text)
  text = re.sub(r"what's", "what is", text) # Present in lots of questions
  text = re.sub(r"who's", "who is", text)
  text = re.sub(r"where's", "where is", text)
  text = re.sub(r"\'s", "", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "cannot ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r" i e ", " ie ", text)
  text = re.sub(r" i.e.", " ie ", text)
  text = re.sub(r" e g ", " eg ", text)
  text = re.sub(r" e.g.", " eg ", text)
  # https://machinelearningmastery.com/clean-text-machine-learning-python/
  tokens = nltk.word_tokenize(text)
  # remove punctuation from each word
  stripped = [w.translate(punctuation_table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if word.isalpha()]
  # filter out stop words
  words = [w for w in words if not w in stop_words]
  # stem
  stemmed = [stemmer.stem(word) for word in words]
  return stemmed

for row in train_data:
  row[3] = clean_text(row[3])
  row[4] = clean_text(row[4])

# https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/


#np.save("dataset/train.npy", train_data)
#np.save("dataset/test.npy", test_data)
#train_data = np.load("dataset/train.npy")
#test_data = np.load("dataset/test.npy")
"""

