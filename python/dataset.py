import numpy as np
import csv

import os
import codecs
import re
import string
import nltk
from keras_bert import Tokenizer

"""
Methods for input preprocessing, before it is ready to be fed into BERT.
"""

punctuation_table = str.maketrans('', '', string.punctuation)
# https://www.kaggle.com/currie32/the-importance-of-cleaning-text
# NLTK seems to have too many, esp the negation ones are essential.
stop_words = [
    "a", "i",
    "an", "as", "by", "at", "if", "in", "is", "of", "on", "so", "to",
    "the", "and", "but", "for",
    "this", "that", "these", "those", "then", "than",
    "what", "which", "while", "who",
    "because", "through", "during", "just", "about",
]

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
  #stemmed = [stemmer.stem(word) for word in words]
  return " ".join(words)

def read_csv(filename, is_test):
  with open(filename, 'r') as f:
      it = csv.reader(f, delimiter = ',', quotechar = '"')
      if is_test:
        data = [[data[1], data[2]] for data in it]
      else:
        data = [[data[3], data[4], data[5]] for data in it]
  return data[1:] # Skip the first line

MODEL_DIR = "model/uncased_L-12_H-768_A-12/"
DICT_PATH = MODEL_DIR + "vocab.txt"

def prepare_data(data, is_test):
  token_dict = {}
  with codecs.open(DICT_PATH, 'r', 'utf8') as reader:
      for line in reader:
          token = line.strip()
          token_dict[token] = len(token_dict)
  tokenizer = Tokenizer(token_dict)
  indices = []
  segments = []
  results = []
  i = 0
  for row in data:
    text1 = clean_text(row[0])
    text2 = clean_text(row[1])
    # In training set, max len is 201. Idk yet what max token count is in test set.
    # Still, pretrained BERT has width of 512, so that's what we will use.
    row_indices, row_segments = tokenizer.encode(first=text1, second=text2, max_len=512)
    indices.append(row_indices)
    segments.append(row_segments)
    #print(tokenizer.tokenize(text1))
    #print(tokenizer.tokenize(text2))
    #print(row_indices)
    #print(row_segments)
    if not is_test:
      results.append(row[2])
    if i % 100 is 0:
      print("i=", i)
    i += 1
  print("Num rows processed: ", i)
  if is_test:
    return np.array(indices), np.array(segments)
  else:
    return np.array(indices), np.array(segments), np.array(results, dtype="float32")

def load_test_data(test_samples):
  if (not os.path.isfile('npydata/test_indices-{}.npy'.format(test_samples)) or
      not os.path.isfile('npydata/test_segments-{}.npy'.format(test_samples))):
    input_data = read_csv("dataset/test.csv", True)
    test_indices, test_segments = prepare_data(input_data[:test_samples], True)
    np.save("npydata/test_indices-{}.npy".format(test_samples), test_indices)
    np.save("npydata/test_segments-{}.npy".format(test_samples), test_segments)
  test_indices = np.load("npydata/test_indices-{}.npy".format(test_samples))
  test_segments = np.load("npydata/test_segments-{}.npy".format(test_samples))
  return test_indices, test_segments

def load_train_data(train_samples, val_samples):
  train_samples = min(train_samples, 380000)
  val_samples = min(val_samples, 20000)
  if (not os.path.isfile('npydata/train_indices-{}.npy'.format(train_samples)) or
      not os.path.isfile('npydata/train_segments-{}.npy'.format(train_samples)) or
      not os.path.isfile('npydata/train_results-{}.npy'.format(train_samples)) or
      not os.path.isfile('npydata/val_indices-{}.npy'.format(val_samples)) or
      not os.path.isfile('npydata/val_segments-{}.npy'.format(val_samples)) or
      not os.path.isfile('npydata/val_results-{}.npy'.format(val_samples))):
    input_data = read_csv("dataset/train.csv", False)
    train_indices, train_segments, train_results = prepare_data(input_data[:train_samples], False)
    val_indices, val_segments, val_results = prepare_data(input_data[-val_samples:], False)
    np.save("npydata/train_indices-{}.npy".format(train_samples), train_indices)
    np.save("npydata/train_segments-{}.npy".format(train_samples), train_segments)
    np.save("npydata/train_results-{}.npy".format(train_samples), train_results)
    np.save("npydata/val_indices-{}.npy".format(val_samples), val_indices)
    np.save("npydata/val_segments-{}.npy".format(val_samples), val_segments)
    np.save("npydata/val_results-{}.npy".format(val_samples), val_results)
  train_indices = np.load("npydata/train_indices-{}.npy".format(train_samples))
  train_segments = np.load("npydata/train_segments-{}.npy".format(train_samples))
  train_results = np.load("npydata/train_results-{}.npy".format(train_samples))
  val_indices = np.load("npydata/val_indices-{}.npy".format(val_samples))
  val_segments = np.load("npydata/val_segments-{}.npy".format(val_samples))
  val_results = np.load("npydata/val_results-{}.npy".format(val_samples))
  return train_indices, train_segments, train_results, val_indices, val_segments, val_results

