import os
import random
import jsonlines
import numpy as np

import torch
from torch.utils.data import Dataset

class VideoFeatureDataset(Dataset):
    def __init__(self, data_path, meta_file, num_frame, frame_skip, null_text, null_text_prob):
        super().__init__()
        data_list = []
        with jsonlines.open(meta_file, 'r') as reader:
            for data in reader:
                if data['num_frame'] >= num_frame * (frame_skip + 1):
                    data_list.append(data)
        self.data_path = data_path
        self.data_list = data_list
        self.num_frame = num_frame
        self.frame_skip = frame_skip
        self.null_text = torch.load(null_text)
        self.null_text_prob = null_text_prob

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        vae_feature = torch.load(os.path.join(self.data_path, self.data_list[idx]['vae_feature']))
        if random.random() < self.null_text_prob:
            text_feature = self.null_text
        else:
            text_feature = torch.load(os.path.join(self.data_path, self.data_list[idx]['text_feature']))
        vae_mean, vae_std = vae_feature['vae_mean'], vae_feature['vae_std']
        sidx = random.randint(0, vae_mean.shape[0] - self.num_frame * (self.frame_skip + 1) + 1)
        eidx = sidx + self.num_frame * (self.frame_skip + 1) - 1
        vae_mean, vae_std = vae_mean[sidx:eidx:self.frame_skip+1], vae_std[sidx:eidx:self.frame_skip+1]
        text_embed, text_mask = text_feature['text_embed'], text_feature['text_mask']
        return vae_mean, vae_std, text_embed, text_mask
