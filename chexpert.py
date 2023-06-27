import re

import pandas as pd
import torch
from torch.utils.data import Dataset


class CheXpertDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.patient_id_to_meta_info = {}
        pattern = r"patient(\d+)"  # extract patient id
        for idx, d in data.iterrows():
            patient_id = re.search(pattern, d.Path).group(1)
            path = f"data/chexpert/{d.Path.replace('CheXpert-v1.0/valid/', 'val/')}"
            if patient_id in self.patient_id_to_meta_info:  # another view of an existing patient
                assert path not in self.patient_id_to_meta_info[patient_id]['image_paths']
                self.patient_id_to_meta_info[patient_id]['image_paths'].append(path)
            else:  # First time patient
                diseases = {}
                # Hardcoded for robustness and consistency
                disease_keys = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
                for key, value in d.iteritems():
                    if key in disease_keys:
                        diseases[key] = value
                disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
                self.patient_id_to_meta_info[patient_id] = {'image_paths': [path], 'disease_keys': disease_keys, 'disease_values': disease_values}
        self.patient_id_to_meta_info = sorted(self.patient_id_to_meta_info.items())

    def __len__(self):
        return len(self.patient_id_to_meta_info)

    def __getitem__(self, idx):
        patient_id, meta_info = self.patient_id_to_meta_info[idx]
        image_paths = meta_info['image_paths']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_paths, labels, keys
