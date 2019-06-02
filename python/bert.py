# https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/load_and_extract.py

import sys
import codecs
import numpy as np
import keras
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import os

import dataset
#import python.dataset as dataset

MODEL_DIR = "model/uncased_L-12_H-768_A-12/"
CONFIG_PATH = MODEL_DIR + "bert_config.json"
CHECKPOINT_PATH = MODEL_DIR + "bert_model.ckpt"
DICT_PATH = MODEL_DIR + "vocab.txt"

BATCH_SIZE = 512
EPOCHS = 10

def prepare_data(data):
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
    text1 = row[3]
    #tokens1 = tokenizer.tokenize(text1)
    text2 = row[4]
    #tokens2 = tokenizer.tokenize(text2)
    row_indices, row_segments = tokenizer.encode(first=text1, second=text2, max_len=512)
    indices.append(row_indices)
    segments.append(row_segments)
    results.append(row[5])
    if i % 100 is 0:
      print("i=", i)
    i += 1
  return np.array(indices), np.array(segments), np.array(results)

def prepare_model():
  bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH)
  bert_model.summary()

  bert_output_shape = bert_model.output.shape.as_list()
  num_bert_outputs = bert_output_shape[1] * bert_output_shape[2]

  top_model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=bert_output_shape[1:]),
    # https://github.com/keras-team/keras/issues/4978#issuecomment-303985365
    keras.layers.Lambda(lambda x: x, output_shape=lambda s:s),
    keras.layers.Reshape((256 * 512,)),
    keras.layers.Dense(1, activation='softmax')
    #keras.layers.Dense(256, activation='relu', input_shape=bert_output_shape),
    #keras.layers.Reshape((256*512,)),
    #keras.layers.Dense(1, activation='softmax'),
  ])
  top_model.output_shape
  top_model.summary()

  # https://github.com/keras-team/keras/issues/3465#issuecomment-314633196
  model = keras.models.Model(inputs=bert_model.input, outputs=top_model(bert_model.output))
  model.compile(
      #loss='categorical_crossentropy',
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])
  model.summary()
  return model

TRAIN_COUNT = 4096
VAL_COUNT = 512

#TRAIN_COUNT = 400000
#VAL_COUNT = 4302

if (not os.path.isfile('npydata/train_indices-{}.npy'.format(TRAIN_COUNT)) or
    not os.path.isfile('npydata/train_segments-{}.npy'.format(TRAIN_COUNT)) or
    not os.path.isfile('npydata/train_results-{}.npy'.format(TRAIN_COUNT)) or
    not os.path.isfile('npydata/val_indices-{}.npy'.format(VAL_COUNT)) or
    not os.path.isfile('npydata/val_segments-{}.npy'.format(VAL_COUNT)) or
    not os.path.isfile('npydata/val_results-{}.npy'.format(VAL_COUNT))):
  input_data = dataset.load_data()[0]
  train_indices, train_segments, train_results = prepare_data(input_data[:TRAIN_COUNT])
  val_indices, val_segments, val_results = prepare_data(input_data[:VAL_COUNT])
  np.save("npydata/train_indices-{}.npy".format(TRAIN_COUNT), train_indices)
  np.save("npydata/train_segments-{}.npy".format(TRAIN_COUNT), train_segments)
  np.save("npydata/train_results-{}.npy".format(TRAIN_COUNT), train_results)
  np.save("npydata/val_indices-{}.npy".format(VAL_COUNT), val_indices)
  np.save("npydata/val_segments-{}.npy".format(VAL_COUNT), val_segments)
  np.save("npydata/val_results-{}.npy".format(VAL_COUNT), val_results)

train_indices = np.load("npydata/train_indices-{}.npy".format(TRAIN_COUNT))
train_segments = np.load("npydata/train_segments-{}.npy".format(TRAIN_COUNT))
train_results = np.load("npydata/train_results-{}.npy".format(TRAIN_COUNT))
val_indices = np.load("npydata/val_indices-{}.npy".format(VAL_COUNT))
val_segments = np.load("npydata/val_segments-{}.npy".format(VAL_COUNT))
val_results = np.load("npydata/val_results-{}.npy".format(VAL_COUNT))

model = prepare_model()

#predicts = model.predict(val_inputs)

history = model.fit(
    [train_indices, train_segments], train_results,
    validation_data=([val_indices, val_segments], val_results),
    epochs=EPOCHS, batch_size=BATCH_SIZE)
    #callbacks=[fit_callback, model_checkpoint])

