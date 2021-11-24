from typing import get_args
import tensorflow as tf
import argparse
import numpy as np
import dvc.api


def train():
    print("TensorFlow version: ", tf.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', default=128, type=float)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    args = parser.parse_args()
    
    with dvc.api.open(
        'data/dataset.npz',
        repo='https://github.com/ssuwani/dvc-tutorial',
        mode="rb"
    ) as fd:
        dataset = np.load(fd)
        train_x = dataset["train_x"]
        train_y = dataset["train_y"]
        test_x = dataset["test_x"]
        test_y = dataset["test_y"]

    train_x, test_x = train_x / 255.0, test_x / 255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(args.units, activation='relu'),
      tf.keras.layers.Dropout(args.dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Training...")
    training_history = model.fit(train_x, train_y, epochs=5)
    loss, acc = model.evaluate(test_x, test_y)

    print("Training-Accuracy={}".format(training_history.history['accuracy']))
    print("Training-Loss={}".format(training_history.history['loss']))
    print("Validation-Accuracy={}".format(acc))
    print("Validation-Loss={}".format(loss))

if __name__ == '__main__':
    train()