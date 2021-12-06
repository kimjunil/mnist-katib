from json import load
from typing import get_args
import tensorflow as tf
import argparse
import numpy as np
import dvc.api
import os
from tensorflow.python.lib.io import file_io
from utils import send_message_to_slack
import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', default=128, type=float)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--deploy', default=False, type=bool)
    return parser.parse_args()
  

def load_data():
    # with dvc.api.open(
    #     'data/dataset.npz',
    #     repo='https://github.com/ssuwani/dvc-tutorial',
    #     mode="rb"
    # ) as fd:
    #     dataset = np.load(fd)
    #     train_x = dataset["train_x"]
    #     train_y = dataset["train_y"]
    #     test_x = dataset["test_x"]
    #     test_y = dataset["test_y"]
    # return (train_x, train_y), (test_x, test_y)
    return tf.keras.datasets.mnist.load_data()

def normalize(data):
  return data / 225.0

def get_model(args):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(args.units, activation='relu'),
      tf.keras.layers.Dropout(args.dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  return model

def train():

    args = get_args()
    model = get_model(args)
    
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x, test_x = normalize(train_x), normalize(test_x)
    
    print("Training...")
    training_history = model.fit(train_x, train_y, validation_split=0.2, epochs=args.epochs)
    
    if args.dropout:
      deploy_model(model)

def arg_to_str():
  return "_".join([f"{x[0]}_{x[1]}" for x in vars(args).items()])

def deploy_model(model, args):
  gcp_bucket = os.getenv("GCS_BUCKET")
  bucket_path = os.path.join(gcp_bucket, "mnist_model")
  save_path = f"{arg_to_str(args)}.h5"
  model.save(save_path)

  gs_path = os.path.join(bucket_path, save_path)
  with file_io.FileIO(save_path, mode='rb') as input_file:
    with file_io.FileIO(gs_path, mode='wb+') as output_file:
      output_file.write(input_file.read())

  # slack_url = os.getenv("WEB_HOOK_URL")
  # if slack_url != None:
  #     send_message_to_slack(slack_url, acc, loss, training_time, gs_path)

if __name__ == '__main__':
    train()
