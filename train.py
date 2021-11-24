# from typing import get_args
# import tensorflow as tf
# import argparse
# import numpy as np
# import dvc.api


# def train():
#     print("TensorFlow version: ", tf.__version__)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--units', default=128, type=float)
#     parser.add_argument('--learning_rate', default=0.01, type=float)
#     parser.add_argument('--dropout', default=0.2, type=float)
#     parser.add_argument('--epochs', default=5, type=int)
#     args = parser.parse_args()
    
#     with dvc.api.open(
#         'data/dataset.npz',
#         repo='https://github.com/ssuwani/dvc-tutorial',
#         mode="rb"
#     ) as fd:
#         dataset = np.load(fd)
#         train_x = dataset["train_x"]
#         train_y = dataset["train_y"]
#         test_x = dataset["test_x"]
#         test_y = dataset["test_y"]

#     train_x, test_x = train_x / 255.0, test_x / 255.0
    
#     model = tf.keras.models.Sequential([
#       tf.keras.layers.Flatten(input_shape=(28, 28)),
#       tf.keras.layers.Dense(args.units, activation='relu'),
#       tf.keras.layers.Dropout(args.dropout),
#       tf.keras.layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     print("Training...")
#     training_history = model.fit(train_x, train_y, epochs=5)
#     loss, acc = model.evaluate(test_x, test_y)

#     print("Training-Accuracy={}".format(training_history.history['accuracy']))
#     print("Training-Loss={}".format(training_history.history['loss']))
#     print("Validation-Accuracy={}".format(acc))
#     print("Validation-Loss={}".format(loss))

# if __name__ == '__main__':
#     train()

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_units", type=int, required=True)
    args = parser.parse_args()
    return args


def train(hidden_units):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    inputs = Input(shape=(28, 28))
    x = layers.Flatten()(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    model.fit(train_x, train_y, epochs=3, validation_split=0.2)
    loss, acc = model.evaluate(test_x, test_y)
    print(f"model test-loss={loss:.4f} test-acc={acc:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args.hidden_units)