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

import sys
import numpy as np
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

RANDOM_STATE = 42

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
    args = sys.argv[1:]
    if args[0] == "initial":
        train_initial_models()
    elif args[0] == "final":
        train_final_models()

def get_datasets():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert the labels using one-hot encoding.
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)

def train_final_models():
    print("Training final models")

    (x_train, y_train), (x_test, y_test) = get_datasets()

    # Get two new models of the best two variant models.
    model_softpool = all_models['Variant_Softpool']()
    model_leaky_relu = all_models['Variant_Leaky_ReLU']()

    # Fit both models
    history_softpool = model_softpool.fit(x_train, y_train, 
        batch_size=BATCH_SIZE, epochs=EPOCHS)
    history_leaky_relu = model_leaky_relu.fit(x_train, y_train, 
        batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # Predict y values and evaluate
    loss_softpool, acc_softpool = model_softpool.evaluate(x_test, y_test, verbose=2)
    loss_leaky_relu, acc_leaky_relu = model_leaky_relu.evaluate(x_test, y_test, verbose=2)

    print('\Softpool test accuracy:', acc_softpool)
    print('\Softpool test loss:', loss_softpool)

    print('\Leaky ReLU test accuracy:', acc_leaky_relu)
    print('\Leaky ReLU test loss:', loss_leaky_relu)

    # print('\Validation accuracy:', validation_acc)


def train_initial_models():
    # Load the train and test sets with labels.
    (x_train, y_train) = get_datasets()[0]

    # K-Fold cross validation setup
    k_fold = KFold(n_splits=NUM_K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    timeline : list[dict] = []
    final_accuracies = {}
    best_accuracies = {}
    best_epochs = {}

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

        # Get model metrics from history and plot them.
        metrics_history = metrics.model_metrics(timeline, ['accuracy', 'loss', 'val_loss', 'val_accuracy'])
        plotting.plot_mean_metric(metrics_history['accuracy'], metrics_history['val_accuracy'], model_label, 'accuracy', bottom=0.7)
        plotting.plot_mean_metric(metrics_history['loss'], metrics_history['val_loss'], model_label, 'loss', top=0.7)

        # Get the last and best mean validation accuracy score for both models.
        final_accuracies[model_label] = (metrics_history['accuracy']['mean'][-1], metrics_history['val_accuracy']['mean'][-1])

        best_idx = np.argmax(metrics_history['val_accuracy']['mean'])
        best_epochs[model_label] = best_idx + 1
        best_accuracies[model_label] = (metrics_history['accuracy']['mean'][best_idx], metrics_history['val_accuracy']['mean'][best_idx])

    # Store metrics for later comparison
    metrics.store_final_accuracy(final_accuracies, "last_epoch_accuracies")
    metrics.store_final_accuracy(best_accuracies, "best_epoch_accuracies")
    metrics.store_best_epoch(best_epochs)

if __name__ == "__main__":
    main()