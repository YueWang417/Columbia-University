import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from IrisMatching import *


def identification_evaluation(train_features, test_features, train_labels, test_labels):
    # Compare performance using different metrics ('l1', 'l2', 'cosine') for the nearest centroid classifier.
    metrics = ['l1', 'l2', 'cosine']
    results = {metric: predict_nearestcentroid(train_features, test_features, train_labels, test_labels, metric=metric)[0] for metric in metrics}
    print(f"results: {results}")
    # Perform Fisher Linear Discriminant to reduce feature dimensionality
    # Evaluate the nearest centroid classifier.
    reduced_train, reduced_test = perform_fld(train_features, test_features, train_labels)
    reduced_results = {metric: predict_nearestcentroid(reduced_train, reduced_test, train_labels, test_labels, metric=metric)[0] for metric in metrics}
    print(f"reduced_results: {reduced_results}")
    #  Table 3: store recognition results
    save_results_table(results, reduced_results, "TABLE 3 Recognition Results Using Different Similarity Measures")
    accuracies = []
    dimensions = range(1, 107, 20)
    # Evaluate the accuracy of the nearest centroid classifier with features of varying dimensionality
    for d in dimensions:
        train, test = perform_fld(train_features, test_features,
                                      train_labels, n_components=d)
        accuracy,_,_ = predict_nearestcentroid(train, test,
                                              train_labels, test_labels,
                                              metric='cosine')
        accuracies.append(accuracy)
    # Plot the accuracy against feature dimensionality
    plot_accuracy_vs_dimension(dimensions, accuracies, "Fig. 10. Recognition results using features of different dimensionality.png")
    # Return the reduced feature sets for train and test after FLD
    return reduced_train, reduced_test





def verification_evaluation(reduced_train, reduced_test, train_labels, test_labels):
    # Use the nearest centroid classifier to evaluate the performance with the cosine distance metric
    accuracy, centroids, predict_result = predict_nearestcentroid(reduced_train, reduced_test,
                                                                  train_labels, test_labels,
                                                                  metric='cosine')

    threshold_values_specific = np.array([0.446, 0.472, 0.502])
    # Compute the pairwise distances between each test sample and each class centroid
    #  it is a matrix where  element [i, j] represents the cosine distance between the i-th test sample in reduced_test and the j-th centroid in centroids
    scores = pairwise_distances(reduced_test, centroids, metric='cosine')
    # Save the calculated FMR and FNMR values for the specific thresholds to a table for reporting
    save_fmr_fnmr_table(scores, test_labels, threshold_values_specific,
                        file_name="TABLE 4 False Match and False Nonmatch Rates with Different Threshold Values")
    # Generate and save the ROC curve to visually analyze performance at various thresholds
    threshold_values_range = 0.2 + 0.01 * np.arange(80, step=5)
    plot_roc_curve(scores, test_labels, threshold_values_range, file_name="Fig. 11. Performance of Gabor filters.")

def calculate_fmr_fnmr(scores, test_labels, threshold_values):
    # Initialize lists to store False Match Rate (FMR) and False Non-Match Rate (FNMR) values
    fmr_values = []
    fnmr_values = []
    # Create a boolean array to check whether the position of the true label for each sample is marked as True
    labels = np.zeros_like(scores, dtype=bool)
    # label here indicates which centroid each test sample actually belongs to.
    # Each row in labels corresponds to a test sample and has a True in the column that corresponds to its true class
    # and False everywhere else
    labels[np.arange(len(test_labels)), test_labels] = True

    for thresh in threshold_values:
        tp = (labels & (scores <= thresh)).sum()
        fp = (~labels & (scores <= thresh)).sum()
        fn = (labels & (scores > thresh)).sum()
        tn = (~labels & (scores > thresh)).sum()
        # Calculate the FMR as the ratio of false positives over the sum of false positives and true negatives
        fmr = fp / (fp + tn)
        # Calculate the FNMR as the ratio of false negatives over the sum of false negatives and true positives
        fnmr = fn / (fn + tp)
        fmr_values.append(fmr)
        fnmr_values.append(fnmr)
    # Return the lists of FMR and FNMR values
    return fmr_values, fnmr_values

# Bellowing are the visualization code for Table 3, Table 4, Figure 10 and Figure 11
# visualization code for Table 3
def save_results_table(results, reduced_results, file_name):
    df = pd.DataFrame({
        'Similarity measure': ['L1 distance measure', 'L2 distance measure', 'Cosine similarity measure'],
        'Original feature set': [results['l1'], results['l2'], results['cosine']],
        'Reduced feature set': [reduced_results['l1'], reduced_results['l2'], reduced_results['cosine']]
    })
    df.set_index('Similarity measure', inplace=True)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values,
             colLabels=df.columns,
             rowLabels=df.index,
             cellLoc='center', rowLoc='center',
             loc='center')
    plt.title(file_name)
    plt.savefig(f"{file_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)

# The visualization code for Figure 10
def plot_accuracy_vs_dimension(dimensions, accuracies, file_name):
    plt.figure(figsize=(8, 6))  # Set the figure size as needed
    plt.plot(dimensions, accuracies, 'k-*', label='Recognition Rate')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate (%)')
    plt.title('Recognition results using features of different dimensionality')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, dpi=300)

# The visualization code for Table 4
def save_fmr_fnmr_table(scores, test_labels, threshold_values, file_name):
    fmr_values, fnmr_values = calculate_fmr_fnmr(scores, test_labels, threshold_values)
    df = pd.DataFrame({
        'Threshold': threshold_values,
        'False match rate (%)': fmr_values,
        'False non-match rate (%)': fnmr_values
    })
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(file_name)
    plt.savefig(file_name + ".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# The visualization code for Figure 11
def plot_roc_curve(scores, test_labels, threshold_values,file_name):
    fmr_values, fnmr_values = calculate_fmr_fnmr(scores, test_labels, threshold_values)
    plt.figure(figsize=(8, 8))
    plt.plot(fmr_values, fnmr_values, 'r-*', label='ROC Curve', markersize=8)
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('ROC Curve - FMR vs FNMR')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name + ".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    plt.close()


