import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

def get_model():
    model_layers = [
        layers.InputLayer(input_shape=(224, 224, 3)),
        
        # Conv1
        layers.Conv2D(filters=96, kernel_size=7, strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv2
        layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),

        # Conv3
        layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'),
        # Conv4
        layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'),

        # Conv5
        layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),
        layers.Flatten(),


        # Full6
        layers.Dense(4096),
        layers.Dropout(rate=0.5),

        # Full7
        layers.Dense(2048),
        layers.Dropout(rate=0.5),

        # Output
        # TODO: Use NUM_CLASSES value.
        layers.Dense(40)
    ]

    model = Sequential(model_layers)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model


