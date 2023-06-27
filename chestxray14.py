import pandas as pd
import torch
from torch.utils.data import Dataset


class ChestXray14Dataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/chestxray14/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/chestxray14/images_jpg/{d.Path.replace(".png", ".jpg")}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                            'Pneumothorax',
                            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append({'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys
