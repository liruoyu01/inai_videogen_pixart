import jsonlines
import torch
import signal
import math
import os
import random
import pickle
from decord import VideoReader

from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class VideoRawTextFeatDataset(Dataset):
    def __init__(
        self,
        data_dir,
        jsonl_dir,
        pickle_save_dir,
        train_size, 
        resize_list, 
        resize_prob,
        fps_list, 
        random_flip=True, 
        center_crop=False,
        load_meta_from_pickle=False,
        max_num_samples=None,
    ):
        super().__init__()
        assert os.path.exists(data_dir), f'{str(data_dir)} must be a folder containing images'
        assert os.path.exists(jsonl_dir) and jsonl_dir.endswith('.jsonl'), f'{str(jsonl_dir)} must be a jsonl file'

        self.data_dir = data_dir  # dataset base folder dir
        self.jsonl_dir = jsonl_dir  # absolute path to the jsonl file
        self.train_size = train_size
        self.fps_list = fps_list
        self.random_flip = random_flip
        self.center_crop = center_crop
        self.resize_list = resize_list
        self.resize_prob = resize_prob
        self.max_num_samples = max_num_samples

        self.pickle_save_dir = pickle_save_dir
        self.load_meta_from_pickle = load_meta_from_pickle
        self.clip_id_meta = defaultdict(dict)
        self.clip_id_list = []

        self.load_meta()

        self.len_data = len(self.clip_id_list)
        random.shuffle(self.clip_id_list)
    
    def load_meta(self):
        if self.load_meta_from_pickle:
            with open(self.pickle_save_dir, 'rb') as f:
                self.clip_id_meta = pickle.load(f)
                self.clip_id_list = list(self.clip_id_meta.keys())
        else:
            self.parse_meta_from_local()
            
    def process_video(self, video):
        '''
        video:[f, h, w, c]
        resize img to target_size with center crop
        ''' 
        video = video.permute([0, 3, 1, 2]).contiguous()  # f,c,h,w
        old_size = video.shape[2:4]

        # ---------- scale --------------
        T_list = []
        if self.random_flip and random.random() > 0.5:
            T_list.append(T.RandomHorizontalFlip(p=1))
        
        if len(self.resize_list)!=0:
            # get random resize param
            prob = random.random()
            if prob <= self.resize_prob[0]:
                resize_h, resize_w = self.resize_list[0]
            elif prob > self.resize_prob[-2]:
                resize_h, resize_w = self.resize_list[-1]
            else:
                for i in range(1,len(self.resize_prob)-1):
                    if prob>self.resize_prob[i-1] and prob<=self.resize_prob[i]:
                        resize_h, resize_w = self.resize_list[i]

            # 如果原始视频大小小于随机选的resize大小，不进行resize;
            # 但如果原始视频大小小于了crop size，仍要进行resize，但是是resize到crop size
            need_resize=True
            if old_size[0]<resize_h and old_size[1]<resize_w:
                if old_size[0]<self.train_size[2] and old_size[1]<self.train_size[3]:
                    resize_h, resize_w = self.train_size[2:4]
                else:
                    need_resize=False

            if need_resize:
                ratio = max(float(resize_h)/(old_size[0]), float(resize_w)/(old_size[1]) )
                new_size = tuple([math.ceil(i*ratio) for i in old_size])
                T_list.append(T.Resize(new_size, interpolation=InterpolationMode.BICUBIC, antialias=True))

        if self.center_crop:
            T_list.append(T.CenterCrop(self.train_size[2:4]))

        transform = T.Compose(T_list)
        video_new = transform(video)
        
        # random crop
        if not self.center_crop:
            x_min = random.randint(0, video_new.shape[2]-self.train_size[2])
            y_min = random.randint(0, video_new.shape[3]-self.train_size[3])
            video_new = video_new[
                :,:,
                x_min:x_min+self.train_size[2],
                y_min:y_min+self.train_size[3]
            ].contiguous()

        return video_new

    def __len__(self):
        # 382668 in total for pexles
        return self.len_data
    
    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        # Edit from https://github.com/pytorch/vision/blob/d03b776a5cd1f4d125eacf127f95d8571a852137/torchvision/datasets/video_utils.py#L278
        step = original_fps / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            num_frames = int(num_frames)
            return torch.arange(0, num_frames*step, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def parse_meta_from_local(self):

        with open(self.jsonl_dir, 'r') as f:
            for line in jsonlines.Reader(f):
                text_feature_file_name = line['text_feature']
                dataset_name =  text_feature_file_name.split('/')[0].split('_')[0]
                clip_id = text_feature_file_name.split('/')[-1].split('.')[0]

                text_feature_file_dir = os.path.join(self.data_dir, text_feature_file_name)
                if os.path.exists(text_feature_file_dir):
                    self.clip_id_meta[clip_id]['text_feature_dir'] = text_feature_file_dir
                
                video_data_file_dir = os.path.join(self.data_dir, dataset_name + '_clip', clip_id + '.mp4')
                if os.path.exists(video_data_file_dir):
                    self.clip_id_meta[clip_id]['video_dir'] = video_data_file_dir
                
                self.clip_id_meta[clip_id]['num_frame'] = line['num_frame']
                self.clip_id_meta[clip_id]['dataset_name'] = dataset_name

                if self.max_num_samples and len(self.clip_id_meta) == self.max_num_samples:
                    break

        f.close()
        self.clip_id_list = list(self.clip_id_meta.keys())

        # save clip_id_meta in pickel
        with open(self.pickle_save_dir, 'wb') as f:
            pickle.dump(self.clip_id_meta, f)
        f.close()

    def __getitem__(self, idx):
        data_dict = {}
        def took_too_long(signum, frame):
            raise TimeoutError('Load', video_path, 'timeout')
        
        while True:
            nidx = self.clip_id_list[idx]
            try:
                video_path = self.clip_id_meta[nidx]['video_dir']
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                video_folder = os.path.basename(os.path.dirname(video_path))

                signal.signal(signal.SIGALRM, took_too_long)
                signal.setitimer(signal.ITIMER_REAL, 120)
                vr = VideoReader(video_path)
                signal.setitimer(signal.ITIMER_REAL, 0)   

                # check1: non-empty (non-corrupted) video
                if len(vr) < self.train_size[0]:
                    print(video_path, 'is too short, video_len=', len(vr), ', self.train_size[0]=', self.train_size[0])
                    idx = (idx + 1) % self.len_data
                    continue

                # check2: non-empty (non-corrupted) video
                video_fps = vr.get_avg_fps()
                # resample video
                if len(self.fps_list)==0:
                    self.train_fps = video_fps
                else:
                    self.train_fps = min(random.choice(self.fps_list), video_fps)
                # print(f'fps ratio {video_fps}/ {self.train_fps}')

                total_frames = len(vr) * self.train_fps / video_fps
                video_idxs = self._resample_video_idx(total_frames, video_fps, self.train_fps)

                # check3: resampled video has enough length before clipping
                video_len = self.train_size[0]
                if len(video_idxs) < video_len:
                    print(video_path, 'is short for clipping, num_frame:', len(video_idxs))
                    idx = (idx + 1) % self.len_data
                    continue
                    
                start = random.randint(0, len(video_idxs)-video_len)
                video_idxs = video_idxs[start:start+video_len]
                video = torch.tensor(vr.get_batch(video_idxs).asnumpy()) # f*h*w*c

                video = self.process_video(video)   # f * c * h * w
                video = video.float() / 127.5 - 1   # [-1, 1]

                # load and process text embedding
                text_feature_file_dir = self.clip_id_meta[nidx]['text_feature_dir']
                t5_text_embedding = torch.load(text_feature_file_dir)['text_embed']
                text_mask = torch.load(text_feature_file_dir)['text_mask']

                data_dict['id'] = video_id
                data_dict['v_feat_path'] = '%s.pt' % video_id

                data_dict['folder'] = video_folder
                data_dict['video'] = video.permute(1, 0, 2, 3).contiguous() # c * f * h * w
                data_dict['fps'] = float(self.train_fps)
                data_dict['num_frames'] = data_dict['video'].shape[1]
                data_dict['dataset_name'] = self.clip_id_meta[nidx]['dataset_name']

                data_dict['text_embedding'] = t5_text_embedding
                data_dict['text_mask'] = text_mask


                return data_dict
                # data_dict: {'dataset': 'pexles', 'id':xxx, 'folder':xxx, 'video':xxx}
            except Exception as e:
                print("Error", e)
                idx = (idx + 1) % self.len_data
                signal.setitimer(signal.ITIMER_REAL, 0)
                continue
