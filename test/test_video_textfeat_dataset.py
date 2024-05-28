
from data.raw_data.video_textfeat_dataset import VideoRawTextFeatDataset
import os
import torch

data_base_dir = '/ML-A100/team/mm/yanghuan/data'
feature_jsonl = '/ML-A100/team/mm/yanghuan/data/pixart-sigma_hq-video-256x256_llava-text_feature_20240425.jsonl'

pickle_file_name = 'pixart-sigma_hq-video_llava-text_feature_20240425_text_feat_raw_clip.pickle'
base_dir = './save'
os.makedirs(base_dir, exist_ok=True)

pickle_file_dir = os.path.join(base_dir, pickle_file_name)

num_frames = 60
# paraemter copy from pexel_dataset
# [num_frames,3,128,256], [[384, 672]], [1.], [30],
dataset = VideoRawTextFeatDataset(
    data_dir=data_base_dir,
    jsonl_dir=feature_jsonl,
    pickle_save_dir=pickle_file_dir,
    train_size=[num_frames,3,128,128],
    resize_list=[[384, 672]], 
    resize_prob = [1.],
    fps_list=[30],
    is_debug=True
)

print(f'loaded data sample count {len(dataset)}')

for i, data in enumerate(dataset):
     # c * f * h * w
    print(data['video'].shape)
    print(torch.max(data['video']))
    print(torch.min(data['video']))
    print(data['id'])
    print(data['dataset_name'])

    print(data['text_embedding'].shape)
    print(data['text_mask'].shape)
    if i == 3:
        break

video_dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=False,
)

for i, batch in enumerate(video_dataloader):
    print(batch['video'].shape)
    print(torch.max(batch['video']))
    print(torch.min(batch['video']))
    print(batch['id'])

    print(batch['text_embedding'].shape)

    if i ==3:
        break
