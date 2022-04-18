import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

from load_data import load_stanford
from learning_rate_scheduler import halving_scheduler_4
import plotting
from models import stanford40_model

# The epoch at the moment of convergence before overfitting began.
# This is used in a file path so be careful to use two digits: 01, ..., 09, 10, 11, etc.
CHOSEN_EPOCH = 11

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 8

def get_model():
    model_layers = [
        # TODO: do a simple normalisation for each image

        layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1./255),
        
        # Conv1
        layers.Conv2D(filters=96, kernel_size=7, strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv2
        layers.Conv2D(filters=256, kernel_size=5, strides=3, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv3
        layers.Conv2D(filters=512, kernel_size=3, strides=3, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv4
        layers.Conv2D(filters=512, kernel_size=3, strides=3, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv5
        layers.Conv2D(filters=512, kernel_size=3, strides=3, padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),
        layers.Flatten(),

        # Full6
        layers.Dense(80),
        layers.Dropout(rate=0.5),

        # Output
        layers.Dense(40)
    ]

    model = Sequential(model_layers)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model

def train_model():
    # get the actual list of files for train, validation and test from stanford dataset from the file names
    train, validation, _, _ = load_stanford(batch_size=BATCH_SIZE)

    # Save model weights every epoch using a checkpoint callback.
    checkpoint_path = "weights/stanford40-epoch{epoch:04d}"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_freq="epoch",
    )

    # Halve learning rate every 4 epochs using a learning rate scheduler callback.
    lr_callback = tf.keras.callbacks.LearningRateScheduler(halving_scheduler_4)

    # Get the model, fit it to the data, and record the training history.
    model = stanford40_model.get_model()
    history = model.fit(train,
        validation_data=validation, batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[lr_callback, save_callback])

    plotting.plot_history_metric(history, "Stanford 40", "accuracy")
    plotting.plot_history_metric(history, "Stanford 40", "loss")

def test_model():
    _, _, test, classes = load_stanford(batch_size=BATCH_SIZE)

    model = stanford40_model.get_model()

    model.load_weights(f"weights/stanford40-epoch00{CHOSEN_EPOCH}").expect_partial()

    # Get prediction values
    # y_pred = model.predict(s40_test, batch_size=BATCH_SIZE)
    # y_pred = np.argmax(y_pred, 1)

    # Predict y values and evaluate
    loss, acc = model.evaluate(test, verbose=1)