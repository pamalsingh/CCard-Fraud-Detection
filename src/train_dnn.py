# src/train_dnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def build_dnn(input_dim, dropout_rate=0.30, learning_rate=0.001):
    model = Sequential()
    # Use explicit Input layer to avoid Keras warning about input_shape on Dense
    from tensorflow.keras.layers import Input
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.20))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_dnn(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=50,
    batch_size=256,
    learning_rate=0.001,
    dropout_rate=0.30
):
    model = build_dnn(X_train.shape[1], dropout_rate=dropout_rate, learning_rate=learning_rate)

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    fit_kwargs = dict(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.20

    history = model.fit(**fit_kwargs)
    return model, history