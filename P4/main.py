import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# TO DISABLE USE OF THE GPU, UNCOMMENT THIS LINE
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
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
from learning_rate_scheduler import halving_scheduler

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

# Train two models and report their final accuracy and loss metrics
def train_final_models(label_1 = 'Baseline', label_2 = 'Variant_Learning Rate'):
    print("Training final models")

    datasets = get_datasets()

    # Get two new models of the best two variant models.
    train_model(label_2, datasets)
    halving_callback = tf.keras.callbacks.LearningRateScheduler(halving_scheduler)
    # Confusion matrix messes up plotting so we do that last
    train_model(label_1, datasets, callbacks=[halving_callback])
    train_model(label_1, datasets, confusion_matrix=True)


# Train one model and report testing accuracy and loss.
def train_model(label, datasets, callbacks=[], confusion_matrix=False):
    (x_train, y_train), (x_test, y_test) = datasets

    # Get two new models of the best two variant models.
    model = all_models[label]()

    # Fit both models
    history = model.fit(x_train, y_train,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Plot accuracy and loss
    plot_label = label if len(callbacks) == 0 else f"Halved_LR_{label}"

    plotting.plot_history_metric(history, plot_label, 'accuracy', bottom=0.7)
    plotting.plot_history_metric(history, plot_label, 'loss', top=0.7)

    # Optionally plot confusion matrix
    if confusion_matrix == True:
        create_confusion_matrix(model, plot_label)
    
    # Predict y values and evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    print(f'{plot_label} test accuracy:', acc)
    print(f'{plot_label} test loss:', loss)

    folder = "./final_weights"
    model.save_weights(f"{folder}/{plot_label}.ckpt")

    return history

# Plot confusion matrix
def create_confusion_matrix(model : Sequential, model_label):
    x_test, y_test = get_datasets()[1]

    y_pred = model.predict(x_test, BATCH_SIZE)

    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test, 1)

    print(f"Creating confusion matrix for {model_label}")
    plotting.plot_confusion_matrix(y_pred, y_test, model_label, CLASS_NAMES)
    
# Trains all model variants using k-fold cross validation
def train_initial_models():

    # Load the train and test sets with labels.
    x_train, y_train = get_datasets()[0]

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