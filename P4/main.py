import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# TO DISABLE USE OF THE GPU, UNCOMMENT THIS LINE
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import KFold
from tensorflow import keras

from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.callbacks import History
from keras import Sequential

from typing import Callable

import plotting
import models
import metrics
from learning_rate_scheduler import scheduler

# Global variables
NUM_CLASSES = 10
NUM_K_FOLDS = 10
BATCH_SIZE = 32
EPOCHS = 10

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

all_models: dict[str, Callable[[], Sequential]] = {
    'Baseline': models.get_baseline,
    'Variant_Learning Rate': models.get_variant_learning_rate,
    'Variant_Dropout': models.get_variant_dropout,
    'Variant_Softpool': models.get_variant_softpool,
    'Variant_Leaky_ReLU': models.get_variant_activation
}

def main():
    # Load the train and test sets with labels.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert the labels using one-hot encoding.
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    # K-Fold cross validation setup
    k_fold = KFold(n_splits=NUM_K_FOLDS, shuffle=True, random_state=42)

    timeline : list[dict] = []
    final_metrics = {}

    for model_label, model_getter in all_models.items():
        print(f"\nStarting K={NUM_K_FOLDS}-fold validation of {model_label}")

        fold = 0
        for train_indices, validation_indices in k_fold.split(x_train):
            fold += 1
            print(f"\nFold {fold}/{NUM_K_FOLDS}\n")

            # Get training data from split for this fold.
            fold_x_train, fold_y_train = x_train[train_indices], y_train[train_indices]

            # Get validation data from split for this fold.
            fold_x_validation, fold_y_validation = x_train[validation_indices], y_train[validation_indices]

            # Redefine the model for the current fold.
            model = model_getter()
            


            # Train and validate the model.
            history : History = model.fit(fold_x_train, fold_y_train, 
                validation_data=(fold_x_validation, fold_y_validation), 
                batch_size=BATCH_SIZE, epochs=EPOCHS)
            
            # If this is the last fold, save the model weights
            if fold == NUM_K_FOLDS:
                checkpoint_path = f"./training_weights/{model_label}.ckpt"
                print(f"\nSaving weights for {model_label}")
                model.save_weights(checkpoint_path)

            timeline.append(history.history)

        metrics_history = metrics.model_metrics(timeline, ['accuracy', 'loss', 'val_loss', 'val_accuracy'])
        plotting.plot_mean_metric(metrics_history['accuracy'], metrics_history['val_accuracy'], model_label, 'accuracy')
        plotting.plot_mean_metric(metrics_history['loss'], metrics_history['val_loss'], model_label, 'loss')

        final_metrics[model_label] = (metrics_history['accuracy']['mean'][-1], metrics_history['val_accuracy']['mean'][-1])

    metrics.store_final_accuracy(final_metrics)
    # # Create and validate final model using all training data.ds
    # model = get_model()
    # opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # model.compile(optimizer=opt,
    #     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy'])

    # history = model.fit(x_train, y_train, 
    #     batch_size=BATCH_SIZE, epochs=EPOCHS)

    # TODO: Pick final model and test it.
    # y_pred = model.predict(x_test)

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    # print('\Test accuracy:', test_acc)
    # print('\Test loss:', test_loss)

    # print('\Validation accuracy:', validation_acc)

if __name__ == "__main__":
    main()