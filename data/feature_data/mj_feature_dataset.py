import json
import torch
import os
import random
from torch.utils.data import Dataset
from model.utils.decoder_utils import DiagonalGaussianDistribution

class ImageJsonFeatureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_json_dir: str,
        num_samples: int= None,
        load_vae_feat: bool= True,
        vae_feat_name: str= 'vae_feature',
        text_feat_name: str= 'text_feature',
        extra_text_feat: str= None,
        sample_latent: bool=True,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.json_dir = meta_json_dir
        self.load_vae_feat = load_vae_feat
        self.sample_latent = sample_latent

        with open(self.json_dir, 'r') as f:
            json_lines = list(map(json.loads, f))
            print(len(json_lines))
            if num_samples and num_samples <= len(json_lines):
                lines_to_load = json_lines[:num_samples]
            else:
                lines_to_load = json_lines
        
        self.image_meta = lines_to_load
        self.vae_feat_name = vae_feat_name
        self.text_feat_name = text_feat_name
        if extra_text_feat:
            assert(isinstance(extra_text_feat, tuple) and len(extra_text_feat)==2)
            self.extra_text_feat_name , self.sample_prob = extra_text_feat



    def __len__(self):
        return len(self.image_meta)

    def __getitem__(self, index):
        json_content = self.image_meta[index]

        if hasattr(self,'extra_text_feat_name'):
            if random.uniform(0,1) <= self.sample_prob:
                text_feat_name_select = self.extra_text_feat_name
            else:
                text_feat_name_select = self.text_feat_name

            text_feature_path = os.path.join(self.data_dir, json_content[text_feat_name_select])

        else:
            text_feature_path = os.path.join(self.data_dir, json_content[self.text_feat_name])
        img_feature_path = os.path.join(self.data_dir, json_content[self.vae_feat_name])

        assert(os.path.exists(text_feature_path))
        assert(os.path.exists(img_feature_path))

        t5_text_embedding = torch.load(text_feature_path)['text_embed']
        text_mask = torch.load(text_feature_path)['text_mask']

        vae_mean = torch.load(img_feature_path)['vae_mean']
        vae_std = torch.load(img_feature_path)['vae_std']

        if self.sample_latent:
            posterior = DiagonalGaussianDistribution(vae_mean, vae_std)
            latent = posterior.sample()
            return latent, t5_text_embedding, text_mask
        
        else:
            return vae_mean, vae_std, t5_text_embedding, text_mask
