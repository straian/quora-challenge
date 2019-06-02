from bert_layer import BertLayer
[["aa bb cc"]]
import tensorflow as tf

import bert
#from bert import run_classifier
#from bert import optimization
#from bert import tokenization

from bert import run_classifier_with_tfhub
#tf.flags.FLAGS.__delattr__('bert_config_file')
#from bert import extract_features

max_seq_length = 100

train_text = [["aa bb cc"]]
test_text = train_text

def convert_text_to_examples(train_text, train_label):
  return ([[0] * len(train_text)], train_text)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

#tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module()
tokenizer.tokenize("This here's an example of using the BERT tokenizer")

train_label = [[0]]
test_label = [[0]]
#train_examples = convert_text_to_examples(train_text, train_label)
#test_examples = convert_text_to_examples(test_text, test_label)

(train_input_ids, train_input_masks, train_segment_ids, train_labels 
) = extract_features.convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(test_input_ids, test_input_masks, test_segment_ids, test_labels
) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)


in_id = tf.keras.layers.Input(shape=(2 * max_seq_length,), name="input_ids")
in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
bert_inputs = [in_id, in_mask, in_segment]

# Instantiate the custom Bert Layer defined above
bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs)

# Build the rest of the classifier 
dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], 
    train_labels,
    validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
    epochs=1,
    batch_size=32
)
