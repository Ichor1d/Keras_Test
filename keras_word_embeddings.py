import re
import string

import tensorflow as tf
from keras import losses
from keras.layers import Embedding, Dense, TextVectorization, Input, GlobalMaxPooling1D, Conv1D, BatchNormalization, \
    Flatten
from tensorflow.keras.optimizers import SGD


batch_size = 32
max_features = 20000
embedding_dim = 128
sequence_length = 500
number_of_classes = 3


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)


if __name__ == '__main__':
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=1337,
    )

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=1337,
    )

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/test", batch_size=batch_size
    )

    # Let's make a text-only dataset (no labels):
    text_ds = raw_train_ds.map(lambda x, y: x)

    # Let's call `adapt` - fits the Embedding layer on the text_ds data corpus:
    vectorize_layer.adapt(text_ds)

    # Vectorize the data. (Separate input data x - the text - from the labels)
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # A integer input for vocab indices.
    inputs = Input(shape=(None,))

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = Embedding(max_features, embedding_dim)(inputs)

    # Conv1D + global max pooling
    x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = Dense(128, activation="relu")(x)

    predictions = Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, predictions)

    sgd = SGD(learning_rate=0.01, decay=0, momentum=0.9)
    model.compile(loss=losses.SparseCategoricalCrossentropy(),
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(train_ds, epochs=3)
