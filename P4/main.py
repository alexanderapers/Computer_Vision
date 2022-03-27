import tensorflow as tf
from sklearn.model_selection import KFold

from keras.datasets import fashion_mnist
from keras import layers, Model, Input
from keras.utils.np_utils import to_categorical

from models import BaselineModel
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

    print(x_test.shape)

    # # # Convert the labels using one-hot encoding.
    # y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    # y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    # K-Fold cross validation setup
    # k_fold = KFold(n_splits=NUM_K_FOLDS)

    # for train_indices, validation_indices in k_fold.split(x_train):
    #     print('Train: %s | validation: %s' % (train_indices, validation_indices))
    # model = BaselineModel()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # loss = model.compute_loss(x=x_test, y=y_test, y_pred=y_pred)
    # print(f"Total loss: {loss}")

if __name__ == "__main__":
    main()
