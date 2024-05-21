import torch
from data.feature_data.mj_feature_dataset import ImageJsonFeatureDataset

img_dataset_name = 'mj_256'
img_dataset_path = {
    'sigma_512': [
        '/ML-A100/team/mm/yanghuan/data/', 
        '/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-image-512x512_llava-text_feature_20240425.jsonl',
    ],
    'sigma_256':[
        '/ML-A100/team/mm/yanghuan/data/',
        '/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-image-256x256_llava-text_feature_20240425.jsonl'
    ],
    'mj_512': [
        '/ML-A100/team/mm/yanghuan/data/', 
        '/ML-A100/team/mm/yanghuan/data/pixart-sigma_midjourney-peter-image-512x512_user-llava-text_feature_20240430.jsonl', 
        ['llava_text_feature', ('user_text_feature', 0.5)],
    ],
    'mj_256': [
        '/ML-A100/team/mm/yanghuan/data/', 
        '/ML-A100/team/mm/yanghuan/data/pixart-sigma_midjourney-peter-image-256x256_user-llava-text_feature_20240430.jsonl'
    ],
}
assert img_dataset_name in img_dataset_path

data_dir = img_dataset_path[img_dataset_name][0]
meta_json_dir = img_dataset_path[img_dataset_name][1]

mj_feature_dataset = ImageJsonFeatureDataset(
    data_dir, 
    meta_json_dir, 
    num_samples=1024,
    text_feat_name='llava_text_feature', 
    extra_text_feat=('user_text_feature', 0.5),
    sample_latent=False,  # return mean, std, text, mask
)
image_dataloader = torch.utils.data.DataLoader(
    dataset=mj_feature_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=False,
)

for _, batch in enumerate(image_dataloader):
    i_mean, i_std, i_text, i_mask = batch
    print(i_mean.shape)
    print(i_text.shape)
    print(i_mask.shape)
