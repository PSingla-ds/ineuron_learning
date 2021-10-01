from utils.model import model_basic
from utils.all_utils import save_model, save_plot
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
# import seaborn as sns

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format = logging_str)

mnist = tf.keras.datasets.mnist
def dataset_prep(mnist):

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    logging.info(f" X_train_full shape: {X_train_full.shape}")

    X_valid, X_train = X_train_full[:5000]/255, X_train_full[5000:]/255
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test/255

    return X_valid, X_train, y_valid, y_train, X_test

X_valid, X_train, y_valid, y_train, X_test = dataset_prep(mnist)


history = model_basic(X_valid, y_valid, X_train, y_train)
save_model(model_basic, filename='ann.model')
save_plot(history,'ann.png')





