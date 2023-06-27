import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chestxray14 import ChestXray14Dataset
from chexpert import CheXpertDataset
from descriptors import disease_descriptors_chexpert, disease_descriptors_chestxray14
from model import InferenceModel
from utils import calculate_auroc

torch.multiprocessing.set_sharing_strategy('file_system')


def inference_chexpert():
    split = 'test'
    dataset = CheXpertDataset(f'data/chexpert/{split}_labels.csv')  # also do test
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_chexpert)

    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_paths, labels, keys = batch
        image_paths = [Path(image_path) for image_path in image_paths]
        agg_probs = []
        agg_negative_probs = []
        for image_path in image_paths:
            probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
            agg_probs.append(probs)
            agg_negative_probs.append(negative_probs)
        probs = {}  # Aggregated
        negative_probs = {}  # Aggregated
        for key in agg_probs[0].keys():
            probs[key] = sum([p[key] for p in agg_probs]) / len(agg_probs)  # Mean Aggregation

        for key in agg_negative_probs[0].keys():
            negative_probs[key] = sum([p[key] for p in agg_negative_probs]) / len(agg_negative_probs)  # Mean Aggregation

        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_chexpert, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(disease_descriptors_chexpert,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys=keys)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    # evaluation
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_chestxray14():
    dataset = ChestXray14Dataset(f'data/chestxray14/Data_Entry_2017_v2020_modified.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_chestxray14)

    all_labels = []
    all_probs_neg = []
    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_chestxray14, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(disease_descriptors_chestxray14,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys=keys)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chexpert', help='chexpert or chestxray14')
    args = parser.parse_args()

    if args.dataset == 'chexpert':
        inference_chexpert()
    elif args.dataset == 'chestxray14':
        inference_chestxray14()
