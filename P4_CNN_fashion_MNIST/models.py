import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import softpool

LEARNING_RATE = 0.001
VARIANT_LEARNING_RATE = 0.0001

# Default activation function for baseline model
def get_relu():
    return layers.ReLU()

# Get all layers for the sequential model
def get_baseline_layers(activation=get_relu, pooling=layers.MaxPool2D):
    return [
        layers.InputLayer(input_shape=(28, 28, 1)), ## layer 1
        layers.Conv2D(filters=32, kernel_size = (3, 3), strides=(1, 1), padding="same"), ## layer 2
        activation(),
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 3

        layers.BatchNormalization(),

        layers.Conv2D(filters=64, kernel_size = (3, 3), strides=(1, 1), padding="same"), ## layer 4
        activation(),
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 5
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128), ## layer 6
        activation(),
        layers.Dense(10) ## layer 7
    ]

# Get the baseline model
def get_baseline(layers=get_baseline_layers, learning_rate = LEARNING_RATE):
    model = Sequential(layers())
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    return model

# Baseline variant model with alternative learning rate.
def get_variant_learning_rate():
    return get_baseline(learning_rate=VARIANT_LEARNING_RATE)

# Retrieves the layers of the baseline model with dropout layers inserted.
def get_variant_dropout_layers():
    # Insert dropout layers after each unit's activation.
    model_layers = get_baseline_layers()
    model_layers.insert(12, layers.Dropout(rate=0.5))
    model_layers.insert(8, layers.Dropout(rate=0.25))
    model_layers.insert(4, layers.Dropout(rate=0.25))
    return model_layers

# Baseline variant model with dropout layers.
def get_variant_dropout():
    return get_baseline(layers=get_variant_dropout_layers)

# Retrieves the layers of the baseline model with soft pooling instead of max pooling layers.
def get_variant_softpool_layers():
    model_layers = get_baseline_layers(pooling=softpool.SoftPooling2D)
    return model_layers

# Baseline variant model with soft pooling instead of max pooling layers.
def get_variant_softpool():
    return get_baseline(layers=get_variant_softpool_layers)

# Leaky relu activation function
def get_leaky_relu():
    return layers.LeakyReLU(alpha=0.1)

# Retrieves the layers of the baseline model with leaky ReLU activation instead of regular ReLU.
def get_variant_activation_layers():
    model_layers = get_baseline_layers(activation=get_leaky_relu)
    return model_layers
    
# Baseline variant model with leaky ReLU activation.
def get_variant_activation():
    return get_baseline(layers=get_variant_activation_layers)