from math import log, exp

import numpy as np
from sklearn.metrics import roc_auc_score


def cos_sim_to_prob(sim):
    return (sim + 1) / 2  # linear transformation to 0 and 1


def log_prob_to_prob(log_prob):
    return exp(log_prob)


def prob_to_log_prob(prob):
    return log(prob)


def calculate_auroc(all_disease_probs, gt_diseases):
    '''
    Calculates the AUROC (Area Under the Receiver Operating Characteristic curve) for multiple diseases.

    Parameters:
    all_disease_probs (numpy array): predicted disease labels, a multi-hot vector of shape (N_samples, 14)
    gt_diseases (numpy array): ground truth disease labels, a multi-hot vector of shape (N_samples, 14)

    Returns:
    overall_auroc (float): the overall AUROC score
    per_disease_auroc (numpy array): an array of shape (14,) containing the AUROC score for each disease
    '''

    per_disease_auroc = np.zeros((gt_diseases.shape[1],))  # num of diseases
    for i in range(gt_diseases.shape[1]):
        # Compute the AUROC score for each disease
        per_disease_auroc[i] = roc_auc_score(gt_diseases[:, i], all_disease_probs[:, i])

    # Compute the overall AUROC score
    overall_auroc = roc_auc_score(gt_diseases, all_disease_probs, average='macro')

    return overall_auroc, per_disease_auroc
