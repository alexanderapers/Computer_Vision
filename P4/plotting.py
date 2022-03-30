import matplotlib.pyplot as plt
import numpy as np

def plot_mean_metric(training_metrics, validation_metrics, model_label, metric_label, bottom=0.0, top=1.0): 
    epochs = range(0,len(training_metrics['mean']))

    # Get all mean/std lists for training and validation
    training_mean = training_metrics['mean']
    training_std = training_metrics['std']
    validation_mean = validation_metrics['mean']
    validation_std = validation_metrics['std']

    # Plot two errorbar plots using the means and standard deviations.
    plt.errorbar(epochs, training_mean, training_std, capsize=4)
    plt.errorbar(epochs, validation_mean, validation_std, capsize=4)

    # Plot labeling and formatting.
    plt.title(f'{model_label} {metric_label}')
    plt.ylabel(metric_label)
    plt.xlabel('epoch')
    plt.xticks(epochs, labels=[f'{i + 1}' for i in epochs])
    plt.ylim(bottom=bottom, top=top)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(visible=True, which='major', axis='y')

    # Save plot and clear pyplot figure and axes for next time.
    plt.savefig(f"metrics/{model_label}_{metric_label}")
    plt.clf()
    plt.cla()

def plot_history_metric(history, model_label, metric_label, bottom=0.0, top=1.0):
    
    metric_data = history.history[metric_label]
    val_metric_data = history.history[f"val_{metric_label}"]

    epochs = range(0,len(metric_data))

    # Plot the training and test ("validation") history. 
    plt.plot(metric_data)
    plt.plot(val_metric_data)

    # Plot labeling and formatting.
    plt.title(f'Final {model_label} {metric_label}')
    plt.ylabel(metric_label)
    plt.xlabel('epoch')
    plt.xticks(epochs, labels=[f'{i + 1}' for i in epochs])
    plt.ylim(bottom=bottom, top=top)
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(visible=True, which='major', axis='y')

    # Save plot and clear pyplot figure and axes for next time.
    plt.savefig(f"metrics/final/final_{model_label}_{metric_label}")
    plt.clf()
    plt.cla()