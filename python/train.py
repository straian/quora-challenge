# https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/load_and_extract.py

import keras
import numpy as np
from time import time
from keras.callbacks import TensorBoard
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.python.client import device_lib

import dataset
#import python.dataset as dataset

"""
Fine-tunes top model (Dense layer with a sigmoid output on top of BERT 512x768 output).
Saves checkpoint after each epoch, so that the trained model can be used for predicitons.
"""

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
GPUS = get_available_gpus()
print("GPU count: ", GPUS)

BATCH_SIZE = (32 * GPUS) if GPUS else 4
EPOCHS = 5

#TRAIN_SAMPLES = 4096 if GPUS else 4
#VAL_SAMPLES = 512 if GPUS else 4
TRAIN_SAMPLES = 380000 if GPUS else 20
VAL_SAMPLES = 20000 if GPUS else 10

MODEL_DIR = "model/uncased_L-12_H-768_A-12/"
CONFIG_PATH = MODEL_DIR + "bert_config.json"
CHECKPOINT_PATH = MODEL_DIR + "bert_model.ckpt"
LOG_PATH = "logs" if GPUS else "logs/logs-local"

def prepare_model():
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

  top_model_dense = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(num_bert_outputs,))
  ])
  top_model_dense.output_shape
  top_model_dense.summary()

  top_model = keras.models.Model(inputs=top_model_flatten.input, outputs=top_model_dense(top_model_flatten.output))
  top_model.output_shape
  top_model.summary()

  # https://github.com/keras-team/keras/issues/3465#issuecomment-314633196
  model = keras.models.Model(inputs=bert_model.input, outputs=top_model(bert_model.output))
  # Default learning rate is 0.001. Decrease it to prevent vanishing gratient (predictions all go to 0) because of sigmoid loss.
  # https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b
  # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  optimizer = keras.optimizers.Adam(lr=0.0003)

  model.compile(
      loss='binary_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'])
  model.summary()
  return model, top_model_dense

print("Preparing model...")
# 'model' is used to train. 'top_model_dense' is only used for checkpointing.
model, top_model_dense = prepare_model()
print("prepared model")

print("Loading data...")
train_indices, train_segments, train_results, val_indices, val_segments, val_results = dataset.load_train_data(TRAIN_SAMPLES, VAL_SAMPLES)
print("Loaded data")

tensorboard = TensorBoard(log_dir=LOG_PATH+"/{}".format(time()))

# Prints some predicted vs actual results, called after each epoch.
def run_validation(epoch, logs):
  MAX_PRINT = 100
  global model
  a = model.predict([val_indices[:MAX_PRINT], val_segments[:MAX_PRINT]]).reshape(min(VAL_SAMPLES, MAX_PRINT))
  b = val_results[:MAX_PRINT]
  combined = np.array([a, b]).transpose()
  print("Predicted vs actual: ", combined.tolist())
val_cb = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: run_validation(epoch, logs))

# Checkpoints top dense layer, called after each epoch.
def checkpoint_top_model_dense(epoch, logs):
  val_loss = logs["val_loss"]
  val_acc = logs["val_acc"]
  global top_model_dense
  keras.models.save_model(
      top_model_dense,
      "checkpoints/top-model-dense.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5".format(epoch=epoch, val_loss=val_loss, val_acc=val_acc))
checkpoint_cb_small = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: checkpoint_top_model_dense(epoch, logs))
checkpoint_cb = keras.callbacks.ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5", save_best_only=True)

run_validation(None, None)

history = model.fit(
    [train_indices, train_segments], train_results,
    validation_data=([val_indices, val_segments], val_results),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb_small, checkpoint_cb, val_cb, tensorboard])

