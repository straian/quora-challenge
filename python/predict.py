from __future__ import absolute_import, division, print_function

import tensorflow as tf
import math
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

DENSE_MODEL_NAME = "top-model-dense.00-0.894-0.729.hdf5"

bert_model = load_trained_model_from_checkpoint(CONFIG_PATH, CHECKPOINT_PATH)
bert_model.summary()
bert_output_shape = bert_model.output.shape.as_list()
num_bert_outputs = bert_output_shape[1] * bert_output_shape[2]

top_model_flatten = keras.Sequential([
  # Need lambda because flatten does not support masking
  # https://github.com/keras-team/keras/issues/4978#issuecomment-303985365
  keras.layers.Lambda(lambda x: x, output_shape=lambda s:s, input_shape=bert_output_shape[1:]),
  keras.layers.Flatten(),
])
top_model_flatten.output_shape
top_model_flatten.summary()

top_model_dense = keras.models.load_model("model/top_model/" + DENSE_MODEL_NAME)
top_model_dense.output_shape
top_model_dense.summary()

top_model = keras.models.Model(inputs=top_model_flatten.input, outputs=top_model_dense(top_model_flatten.output))
top_model.output_shape
top_model.summary()

model = keras.models.Model(inputs=bert_model.input, outputs=top_model(bert_model.output))
model.summary()

#TEST_SAMPLES = 2345796
TEST_SAMPLES = 32

# Only predict one out of each DOWN_SAMPLE_RATE sample.
#DOWN_SAMPLE_RATE = 1000
DOWN_SAMPLE_RATE = 1

test_indices0, test_segments0 = dataset.load_test_data(TEST_SAMPLES)
i = 0
test_indices = []
test_segments = []
for i in range(int(math.floor(TEST_SAMPLES / DOWN_SAMPLE_RATE))):
  test_indices.append(test_indices0[i * DOWN_SAMPLE_RATE])
  test_segments.append(test_segments0[i * DOWN_SAMPLE_RATE])
  i += DOWN_SAMPLE_RATE

print("Starting prediction: ", len(test_indices))
outputs = model.predict([test_indices, test_segments], verbose=1)
print("Ended prediction: ", len(outputs))
lines = [[i * DOWN_SAMPLE_RATE, int(round(outputs[i][0]))] for i in range(len(outputs))]

with open('outputs/submittable-{}-{}.csv'.format(TEST_SAMPLES, DOWN_SAMPLE_RATE), 'w') as write_file:
  writer = csv.writer(write_file)
  writer.writerow(["test_id" , "is_duplicate"])
  writer.writerows(lines)

