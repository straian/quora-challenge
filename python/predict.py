from __future__ import absolute_import, division, print_function

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
from tensorflow.python.client import device_lib
import csv
import numpy as np

import dataset
#import python.dataset as dataset

"""
Loads checkpoint model and predicts results for test data.
"""

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
GPUS = get_available_gpus()
print("GPU count: ", GPUS)

MODEL_DIR = "model/uncased_L-12_H-768_A-12/"
CONFIG_PATH = MODEL_DIR + "bert_config.json"
CHECKPOINT_PATH = MODEL_DIR + "bert_model.ckpt"
LOG_PATH = "logs" if GPUS else "logs/logs-local"

MODEL_GPU_NAME = "weights.05-0.586.hdf5"
MODEL_LOCAL_NAME = "weights.00-1.000-1.000.hdf5"

model = None
if GPUS:
  # https://github.com/CyberZHG/keras-bert/issues/66
  model = keras.models.load_model("model/top_model/" + MODEL_GPU_NAME, custom_objects=get_custom_objects())
else:
  top_model = keras.models.load_model("model/top_model/" + MODEL_LOCAL_NAME)
  top_model.output_shape
  top_model.summary()

  bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH)
  bert_model.summary()
  bert_output_shape = bert_model.output.shape.as_list()
  num_bert_outputs = bert_output_shape[1] * bert_output_shape[2]

  model = keras.models.Model(inputs=bert_model.input, outputs=top_model(bert_model.output))
model.summary()

TEST_SAMPLES = 2345796
#TEST_SAMPLES = 32

test_indices, test_segments = dataset.load_test_data(TEST_SAMPLES)

print("Starting prediction: ", len(test_indices))
outputs = model.predict([test_indices, test_segments])
print("Ended prediction: ", len(outputs))
lines = [[i, int(round(outputs[i][0]))] for i in range(len(outputs))]

with open('outputs/submittable-{}.csv'.format(TEST_SAMPLES), 'w') as write_file:
  writer = csv.writer(write_file)
  writer.writerow(["test_id" , "is_duplicate"])
  writer.writerows(lines)

