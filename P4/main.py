import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# TO DISABLE THE GPU, UNCOMMENT THIS LINE
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

from keras.datasets import fashion_mnist
from keras import layers, Sequential
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt



# Global variables
NUM_CLASSES = 10
NUM_K_FOLDS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 2

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_model():
    model = Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)), ## layer 1
        layers.Conv2D(filters=32, kernel_size = (3, 3), strides=(1, 1), padding="same", activation='relu'), ## layer 2
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 3
        layers.LayerNormalization(),

        layers.Conv2D(filters=64, kernel_size = (3, 3), strides=(1, 1), padding="same", activation='relu'), ## layer 4
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 5
        layers.LayerNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'), ## layer 6
        layers.Dense(10) ## layer 7
    ])
    return model

def main():
    # Load the train and test sets with labels.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # # do we need to shuffle the data before? is ti ordered on label?
    # x_train, x_validation, y_train, y_validation = train_test_split(
    #                 x_train, y_train, test_size = 0.2, random_state = 0, shuffle = True)

    #print(x_validation.shape)

    # Convert the labels using one-hot encoding.
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    # y_validation = to_categorical(y_validation, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    # K-Fold cross validation setup
    k_fold = KFold(n_splits=NUM_K_FOLDS, shuffle=True, random_state=42)

    timeline = []

    for train_indices, validation_indices in k_fold.split(x_train):

        # Get training data from split for this fold.
        fold_x_train, fold_y_train = x_train[train_indices], y_train[train_indices]

        # Get validation data from split for this fold.
        fold_x_validation, fold_y_validation = x_train[validation_indices], y_train[validation_indices]

        # Redefine the model for the current fold.
        model = get_model()
        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt,
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        # Train and validate the model.
        history = model.fit(fold_x_train, fold_y_train, 
            validation_data=(fold_x_validation, fold_y_validation), 
            batch_size=BATCH_SIZE, epochs=EPOCHS)
        
        timeline.append(history)

    
    # Create and validate final model using all training data.
    model = get_model()
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
        batch_size=BATCH_SIZE, epochs=EPOCHS)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # TODO: Pick final model and test it.
    # y_pred = model.predict(x_test)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\Test accuracy:', test_acc)
    print('\Test loss:', test_loss)

    # print('\Validation accuracy:', validation_acc)

if __name__ == "__main__":
    main()
