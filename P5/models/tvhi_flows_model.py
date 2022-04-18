import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


def get_model():
    model_layers = [

        layers.InputLayer(input_shape=(16, 224, 224, 2)),
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

        # # Full6
        # layers.Dense(4096),
        # layers.Dropout(rate=0.5),

        # Full7

        layers.Dense(100),
        layers.Dropout(rate=0.5),

        # Output
        # TODO: Use NUM_CLASSES value.
        layers.Dense(4)
    ]

    model = Sequential(model_layers)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


