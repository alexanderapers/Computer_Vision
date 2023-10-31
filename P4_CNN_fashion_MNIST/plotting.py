import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# From https://www.kaggle.com/code/grfiv4/plot-a-confusion-matrix/notebook
def __plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    # plt.show()


def plot_confusion_matrix(y_test, y_pred, model_label, display_labels):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    __plot_confusion_matrix(cm, display_labels, title=f"{model_label} Confusion Matrix")
    # Save plot and clear pyplot figure and axes for next time.
    plt.savefig(f"metrics/final/final_{model_label}_confusion", bbox_inches = "tight")
    plt.clf()
    plt.cla()