# https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/load_and_extract.py

import os
import sys
import codecs
import keras
import numpy as np
from time import time
from keras.callbacks import TensorBoard
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.python.client import device_lib

import dataset
#import python.dataset as dataset

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

GPUS = get_available_gpus()

print("GPU count: ", GPUS)

BATCH_SIZE = (128 * GPUS) if GPUS else 4
EPOCHS = 5

TRAIN_SAMPLES = 40960 if GPUS else 16
VAL_SAMPLES = 5120 if GPUS else 8
#TRAIN_SAMPLES = 400000 if GPUS else 20
#VAL_SAMPLES = 4032 if GPUS else 10

MODEL_DIR = "model/uncased_L-12_H-768_A-12/"
CONFIG_PATH = MODEL_DIR + "bert_config.json"
CHECKPOINT_PATH = MODEL_DIR + "bert_model.ckpt"
DICT_PATH = MODEL_DIR + "vocab.txt"
LOG_PATH = "logs" if GPUS else "logs/logs-local"

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
    text1 = dataset.clean_text(row[0])
    text2 = dataset.clean_text(row[1])
    # In training set, max len is 201. Idk yet what max token count is in test set.
    # Still, pretrained BERT has width of 512, so that's what we will use.
    row_indices, row_segments = tokenizer.encode(first=text1, second=text2, max_len=512)
    indices.append(row_indices)
    segments.append(row_segments)
    #print(tokenizer.tokenize(text1))
    #print(tokenizer.tokenize(text2))
    #print(row_indices)
    #print(row_segments)
    results.append(row[2])
    if i % 100 is 0:
      print("i=", i)
    i += 1
  return np.array(indices), np.array(segments), np.array(results, dtype="float32")

def prepare_model():
  bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH)
  bert_model.summary()
  bert_output_shape = bert_model.output.shape.as_list()
  num_bert_outputs = bert_output_shape[1] * bert_output_shape[2]
  top_model = keras.Sequential([
    # https://github.com/keras-team/keras/issues/4978#issuecomment-303985365
    keras.layers.Lambda(lambda x: x, output_shape=lambda s:s, input_shape=bert_output_shape[1:]),
    keras.layers.Flatten(),
    #keras.layers.Dense(256, activation='relu', input_shape=bert_output_shape[1:]),
    keras.layers.Dense(1, activation='sigmoid')
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

def len_input(segment):
  li = segment.tolist()
  # https://stackoverflow.com/questions/6890170/how-to-find-the-last-occurrence-of-an-item-in-a-python-list
  return len(li) - li[::-1].index(1)
#lens = [len_input(s) for s in train_segments]

if (not os.path.isfile('npydata/train_indices-{}.npy'.format(TRAIN_SAMPLES)) or
    not os.path.isfile('npydata/train_segments-{}.npy'.format(TRAIN_SAMPLES)) or
    not os.path.isfile('npydata/train_results-{}.npy'.format(TRAIN_SAMPLES)) or
    not os.path.isfile('npydata/val_indices-{}.npy'.format(VAL_SAMPLES)) or
    not os.path.isfile('npydata/val_segments-{}.npy'.format(VAL_SAMPLES)) or
    not os.path.isfile('npydata/val_results-{}.npy'.format(VAL_SAMPLES))):
  input_data = dataset.load_data()[0]
  train_indices, train_segments, train_results = prepare_data(input_data[:TRAIN_SAMPLES])
  val_indices, val_segments, val_results = prepare_data(input_data[-VAL_SAMPLES:])
  np.save("npydata/train_indices-{}.npy".format(TRAIN_SAMPLES), train_indices)
  np.save("npydata/train_segments-{}.npy".format(TRAIN_SAMPLES), train_segments)
  np.save("npydata/train_results-{}.npy".format(TRAIN_SAMPLES), train_results)
  np.save("npydata/val_indices-{}.npy".format(VAL_SAMPLES), val_indices)
  np.save("npydata/val_segments-{}.npy".format(VAL_SAMPLES), val_segments)
  np.save("npydata/val_results-{}.npy".format(VAL_SAMPLES), val_results)

train_indices = np.load("npydata/train_indices-{}.npy".format(TRAIN_SAMPLES))
train_segments = np.load("npydata/train_segments-{}.npy".format(TRAIN_SAMPLES))
train_results = np.load("npydata/train_results-{}.npy".format(TRAIN_SAMPLES))
val_indices = np.load("npydata/val_indices-{}.npy".format(VAL_SAMPLES))
val_segments = np.load("npydata/val_segments-{}.npy".format(VAL_SAMPLES))
val_results = np.load("npydata/val_results-{}.npy".format(VAL_SAMPLES))

model = prepare_model()

tensorboard = TensorBoard(log_dir=LOG_PATH+"/{}".format(time()))

def run_validation(epoch, logs):
  MAX_PRINT = 100
  global model
  a = model.predict([val_indices[:MAX_PRINT], val_segments[:MAX_PRINT]]).reshape(min(VAL_SAMPLES, MAX_PRINT))
  b = val_results[:MAX_PRINT]
  combined = np.array([a, b]).transpose()
  print("Predicted vs actual: ", combined.tolist())

val_cb = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: run_validation(epoch, logs))

run_validation(None, None)

history = model.fit(
    [train_indices, train_segments], train_results,
    validation_data=([val_indices, val_segments], val_results),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[val_cb, tensorboard])

