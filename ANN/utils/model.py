import tensorflow as tf
import logging

def model_basic(X_valid, y_valid, X_train, y_train):
    LAYERS = [
            tf.keras.layers.Flatten(input_shape = (28,28), name = "inputLayer"),
            tf.keras.layers.Dense(300, activation='relu', name = 'Hiddenlayer1'),
            tf.keras.layers.Dense(100, activation='relu', name = 'Hiddenlayer2'),
            tf.keras.layers.Dense(10, activation='softmax', name = 'OutputLayer')
    ]

    model_clf = tf.keras.models.Sequential(LAYERS)

    logging.info(f"model summary {model_clf.summary}")

    LOSS_FUNCTION = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    model_clf.compile(loss = LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    EPOCHS = 30
    VALIDATION = (X_valid, y_valid)
    history = model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)
    
    return history

