"""Neural-network model factories for the complaint classifier."""

from __future__ import annotations


def build_recurrent_model(
    architecture: str, vocabulary_size: int, max_length: int, num_classes: int
):
    """Build and compile one of the required RNN, LSTM or GRU classifiers."""
    from tensorflow import keras

    recurrent_layers = {
        "simple_rnn": keras.layers.SimpleRNN,
        "lstm": keras.layers.LSTM,
        "gru": keras.layers.GRU,
    }
    if architecture not in recurrent_layers:
        raise ValueError(f"Unsupported architecture: {architecture}")

    recurrent = recurrent_layers[architecture]
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_length,)),
            keras.layers.Embedding(vocabulary_size, 128, mask_zero=True),
            recurrent(96, dropout=0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name=f"complaint_{architecture}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
