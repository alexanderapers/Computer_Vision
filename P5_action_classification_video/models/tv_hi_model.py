import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers, Model
from learning_rate_scheduler import halving_scheduler_5
from load_data import load_tvhi

import plotting
from models import stanford40_model

# The epoch at the moment of convergence before overfitting began.
# This is used in a file path so be careful to use two digits: 01, ..., 09, 10, 11, etc.
CHOSEN_EPOCH = "08"

# Training parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
EPOCHS = 20

def get_model():

    model = stanford40_model.get_model()
    model.load_weights("weights/stanford40/stanford40-epoch0011").expect_partial()


    for layer in model.layers:
        layer.trainable = False
    
    x = model.layers[-4].output
    x = layers.Dense(30)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(30)(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(4)(x)

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = Model(
        inputs = model.input,
        outputs = predictions)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

def train_model():
    # Halve learning rate every 4 epochs using a learning rate scheduler callback.
    lr_callback = tf.keras.callbacks.LearningRateScheduler(halving_scheduler_5)

    checkpoint_path = "weights/tv-hi/tv-hi-epoch{epoch:04d}"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_freq="epoch",
    )

    (train, _), (validation, _), _ = load_tvhi(batch_size=BATCH_SIZE)
    
    model = get_model()
    history = model.fit(train,
        validation_data=validation, batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback])
        # callbacks=[save_callback, lr_callback])

    plotting.plot_history_metric(history, "TV-HI", "accuracy")
    plotting.plot_history_metric(history, "TV-HI", "loss")

def test_model():
    _, _, (test, _) = load_tvhi(batch_size=BATCH_SIZE)

    model = get_model()

    model.load_weights(f"weights/tv-hi/tv-hi-epoch00{CHOSEN_EPOCH}").expect_partial()

    print(f"\nTesting epoch {CHOSEN_EPOCH}...")
    loss, acc = model.evaluate(test, verbose=1)
