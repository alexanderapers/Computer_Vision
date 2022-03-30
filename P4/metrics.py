import numpy as np
import csv

def calculate_metric_mean_std(timeline, metric_label):
    metric = [history[metric_label] for history in timeline]
    mean_metric = np.mean(metric, axis=0)
    std_metric = np.std(metric, axis=0)
    return mean_metric, std_metric

def model_metrics(timeline, labels):
    metrics_dict = {}

    for label in labels:
        mean_metric, std_metric = calculate_metric_mean_std(timeline, label)
        metrics_dict[label] = {
            'mean': mean_metric,
            'std': std_metric
        }

    return metrics_dict

def store_final_accuracy(final_metrics):
    # open file for writing, "w" is writing
    w = csv.writer(open("output.csv", "w"))

    w.writerow(['Model', 'Mean training accuracy', 'Mean validation accuracy'])

    # loop over dictionary keys and values
    for model_label, accuracies in final_metrics.items():
        training, validation = accuracies
        # write every key and value to file
        w.writerow([model_label, training, validation])
