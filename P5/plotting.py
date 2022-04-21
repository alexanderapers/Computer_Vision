import os
from matplotlib import pyplot as plt

def plot_history_metric(history, model_label, metric_label):
    
    metrics_dir = r"metrics/"
    if not os.path.isdir(metrics_dir):
        os.mkdir(metrics_dir)

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
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(visible=True, which='major', axis='y')

    # Save plot and clear pyplot figure and axes for next time.
    plt.savefig(f"metrics/{model_label}_{metric_label}")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()