import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers, Model
from learning_rate_scheduler import halving_scheduler_10
from load_data import load_tvhi

import plotting
from models import stanford40_model

BATCH_SIZE = 8
EPOCHS = 15

def get_model():

    model = stanford40_model.get_model()
    model.load_weights("weights/stanford40-epoch0011").expect_partial()


    for layer in model.layers:
        layer.trainable = False
    
    x = model.layers[-2].output
    x = layers.Dense(160)(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(4)(x)

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Model(
        inputs = model.input,
        outputs = predictions)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])


    return model


def train_model():

    checkpoint_path = "weights/tv-hi/tv-hi-epoch{epoch:04d}"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_freq="epoch",
    )

    lr_callback = tf.keras.callbacks.LearningRateScheduler(halving_scheduler_10)

    (train, _), (validation, _), (test, _) = load_tvhi()
    
    model = get_model()
    # model.summary()
    history = model.fit(train,
        validation_data=validation, batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback, lr_callback])

    plotting.plot_history_metric(history, "TV-HI", "accuracy")
    plotting.plot_history_metric(history, "TV-HI", "loss")

    loss, acc = model.evaluate(test, verbose=1)