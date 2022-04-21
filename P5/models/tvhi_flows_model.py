import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

import plotting
from load_data import load_tvhi

# The epoch at the moment of convergence before overfitting began.
# This is used in a file path so be careful to use two digits: 01, ..., 09, 10, 11, etc.
CHOSEN_EPOCH = "16"

# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 2
EPOCHS = 20


def get_model():
    model_layers = [

        layers.InputLayer(input_shape=(16, 224, 224, 2), name="flow_input"),
        tf.keras.layers.Rescaling(1. / 255),

        # Conv1
        layers.Conv3D(filters=128, kernel_size=7, strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding="same"),

        # Conv2
        layers.Conv3D(filters=256, kernel_size=5, strides=3, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding="same"),

        # Conv3
        layers.Conv3D(filters=512, kernel_size=3, strides=3, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, padding="same"),

        layers.Flatten(),

        # Full4
        layers.Dense(100),
        layers.Dropout(rate=0.5),

        # Output
        layers.Dense(4)
    ]

    model = Sequential(model_layers)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model():

    checkpoint_path = "weights/tv-hi-flow/tv-hi-flow-epoch{epoch:04d}"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=0,
        save_weights_only=True,
        save_freq="epoch",
    )

    (_, train_flow), (_, validation_flow), _ = load_tvhi(batch_size=BATCH_SIZE)
    
    model = get_model()
    # model.summary()
    history = model.fit(train_flow,
        validation_data=validation_flow, batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback])

    plotting.plot_history_metric(history, "TV-HI flows", "accuracy")
    plotting.plot_history_metric(history, "TV-HI flows", "loss")

def test_model():
    _, _, (_, test_flow) = load_tvhi(batch_size=BATCH_SIZE)

    model = get_model()

    model.load_weights(f"weights/tv-hi-flow/tv-hi-flow-epoch00{CHOSEN_EPOCH}").expect_partial()

    print(f"\nTesting epoch {CHOSEN_EPOCH}...")
    loss, acc = model.evaluate(test_flow, verbose=1)
