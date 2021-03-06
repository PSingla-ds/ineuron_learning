
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format = logging_str)

def main(data, modelName, plotName, eta, epochs):
    """ it is used to generate the dataframe and then split it into dependent and independent variables

    Args:
        data ([type]): [description]
        modelName ([type]): [description]
        plotName ([type]): [description]
        eta ([type]): [description]
        epochs ([type]): [description]
    """
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe {df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)