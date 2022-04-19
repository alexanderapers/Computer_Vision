import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers, Model
from learning_rate_scheduler import halving_scheduler_5
from load_data import load_tvhi
from keras.layers import *

import plotting
from models import tv_hi_model, tvhi_flows_model

CHOSEN_EPOCH = "12" 

# Training parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
EPOCHS = 20


def get_model():
    model_tvhi = tv_hi_model.get_model()
    model_tvhi.load_weights("weights/tv-hi/tv-hi-epoch0008").expect_partial()

    model_tvhi_flows = tvhi_flows_model.get_model()
    model_tvhi_flows.load_weights("weights/tv-hi-flow/tv-hi-flow-epoch0016").expect_partial()

    for layer in model_tvhi.layers:
        layer.trainable = False

    for layer in model_tvhi_flows.layers:
        layer.trainable = False

    x1 = model_tvhi.layers[-4].output
    x2 = model_tvhi_flows.layers[-4].output

    mergedOut = Concatenate()([x1, x2])

    mergedOut = Flatten()(mergedOut)
    mergedOut = Dense(128)(mergedOut)
    mergedOut = Dropout(0.5)(mergedOut)
    mergedOut = Dense(64)(mergedOut)
    mergedOut = Dropout(0.5)(mergedOut)
    predictions = layers.Dense(4)(mergedOut)

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = Model(
        inputs=[model_tvhi.input, model_tvhi_flows.input],
        outputs=predictions)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

# From https://stackoverflow.com/questions/63105401/tensorflow-2-0-create-a-dataset-to-feed-a-model-with-multiple-inputs-of-differen
def __data_tx(d1, d2, t):
    return {"frame_input": d1, "flow_input": d2}, tf.transpose(t)

# Beautifully alters the dataset so the two-stream network will take it.
def __reform_dataset(frames, flows):
    labels = frames.map(lambda x, y: y)
    frames = frames.map(lambda x, y: x)
    flows = flows.map(lambda x, y: x)

    test_dataset = tf.data.Dataset.zip((frames, flows, labels)).map(__data_tx)

    return test_dataset

def train_model():
    # Halve learning rate every 4 epochs using a learning rate scheduler callback.
    lr_callback = tf.keras.callbacks.LearningRateScheduler(halving_scheduler_5)

    checkpoint_path = "weights/tv-hi-two-stream/tv-hi-two-stream-epoch{epoch:04d}"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_freq="epoch",
    )

    (train, train_flows), (val, val_flows), _ = load_tvhi(batch_size=BATCH_SIZE)
    
    # Horribly deform datasets to make them work with the two-stream input.
    train_dataset = __reform_dataset(train, train_flows)
    val_dataset = __reform_dataset(val, val_flows)

    model = get_model()
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=[save_callback])
    # callbacks=[save_callback, lr_callback])

    plotting.plot_history_metric(history, "TV-HI two stream", "accuracy")
    plotting.plot_history_metric(history, "TV-HI two stream", "loss")

def test_model():
    _, _, (test, test_flows) = load_tvhi(batch_size=BATCH_SIZE)
    # x1, x2, y = __deform_dataset(test, test_flows)
    test_dataset = __reform_dataset(test, test_flows)

    model = get_model()

    model.load_weights(f"weights/tv-hi-two-stream/tv-hi-two-stream-epoch00{CHOSEN_EPOCH}").expect_partial()

    print(f"\nTesting epoch {CHOSEN_EPOCH}...")
    loss, acc = model.evaluate(test_dataset, verbose=1)
