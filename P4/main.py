import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow import keras

from keras.datasets import fashion_mnist
from keras import layers, Model, Input
from keras.utils.np_utils import to_categorical

# Global variables
NUM_CLASSES = 10
NUM_K_FOLDS = 10

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



def main():
    # Load the train and test sets with labels.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # do we need to shuffle the data before? is ti ordered on label?
    x_train, x_validation, y_train, y_validation = train_test_split(
                    x_train, y_train, test_size = 0.2, random_state = 0, shuffle = True)

    #print(x_validation.shape)

    # Convert the labels using one-hot encoding.
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_validation = to_categorical(y_validation, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    # K-Fold cross validation setup
    # k_fold = KFold(n_splits=NUM_K_FOLDS)

    # for train_indices, validation_indices in k_fold.split(x_train):
    #     print('Train: %s | validation: %s' % (train_indices, validation_indices))
    # model = BaselineModel()

    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)), ## layer 1
        layers.Conv2D(filters=32, kernel_size = (3, 3), strides=(2, 2), padding="same", activation='relu'), ## layer 2
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 3
        layers.LayerNormalization(),

        layers.Conv2D(filters=64, kernel_size = (3, 3), strides=(1, 1), padding="same", activation='relu'), ## layer 4
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same"), ## layer 5
        layers.LayerNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'), ## layer 6
        layers.Dense(10) ## layer 7
    ])

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    validation_loss, validation_acc = model.evaluate(x_validation,  y_validation, verbose=2)

    print('\Validation accuracy:', validation_acc)

if __name__ == "__main__":
    main()
