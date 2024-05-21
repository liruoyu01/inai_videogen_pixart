import os
import random
import jsonlines

import torch
from torch.utils.data import Dataset

class ImageFeatureDataset(Dataset):
    def __init__(self, data_path, meta_file, null_text, null_text_prob):
        super().__init__()
        data_list = []
        with jsonlines.open(meta_file, 'r') as reader:
            for data in reader:
                data_list.append(data)
        self.data_path = data_path
        self.null_text = torch.load(null_text)
        self.null_text_prob = null_text_prob
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        vae_feature = torch.load(os.path.join(self.data_path, self.data_list[idx]['vae_feature']))
        if random.random() < self.null_text_prob:
            text_feature = self.null_text
        else:
            text_feature = torch.load(os.path.join(self.data_path, self.data_list[idx]['text_feature']))
        vae_mean, vae_std = vae_feature['vae_mean'], vae_feature['vae_std']
        text_embed, text_mask = text_feature['text_embed'], text_feature['text_mask']
        return vae_mean, vae_std, text_embed, text_mask
