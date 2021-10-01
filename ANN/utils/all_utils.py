import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import joblib

def save_model(model, filename):
    """This saves the trained model to
    Args:
        model (python object): trained model to
        filename (str): path to save the trained model
    """
    logging.info("saving the trained model")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    filepath = os.path.join(model_dir, filename)  # model/filename
    joblib.dump(model, filepath)
    logging.info(f"saved the trained model {filepath}")


def save_plot(history, filename):
    def plot_accuracy(history):
        pd.DataFrame(history.history).plot(figsize = (10,7))
        plt.grid(True)

    plot_accuracy(history)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    plotPath = os.path.join(plot_dir, filename)  # model/filename
    plt.savefig(plotPath)