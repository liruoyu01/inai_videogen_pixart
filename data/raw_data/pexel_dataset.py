import jsonlines
import torch
import signal
import math
import os
import random
from decord import VideoReader

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class PexelsDataset(Dataset):
    def __init__(
            self,  
            data_dir, 
            meta_file_dir,
            train_size, 
            resize_list, 
            resize_prob,
            fps_list, 
            random_flip=True, 
            center_crop=False,
            isdebug=False,
            include_caption=False,
        ):
        super().__init__()

        assert os.path.exists(meta_file_dir)
        self.meta_file_dir = meta_file_dir
        self.include_caption = include_caption

        # read meta file and load video paths to self.video_list
        self.read_meta(debug=isdebug)
        if self.include_caption:
            assert len(self.video_list) == len(self.caption_list)
        self.len_data = len(self.video_list)
        assert self.len_data > 0
        self.shuffle_indices = [i for i in list(range(self.len_data))]
        random.shuffle(self.shuffle_indices)

        self.data_dir = data_dir
        self.train_size = train_size
        self.fps_list = fps_list
        self.random_flip = random_flip
        self.center_crop = center_crop
        self.resize_list = resize_list
        self.resize_prob = resize_prob

    def __len__(self):
        # 382668 in total for pexles
        return self.len_data

    def read_meta(self, debug=False):
        video_list = []
        caption_list = []
        with open(self.meta_file_dir, 'r', encoding="utf-8") as f:
            for line in jsonlines.Reader(f):
                video_list.append(line['clip_id']+'.mp4')
                if self.include_caption:
                    caption_list.append(str(line['llava_medium']).strip())

        # for debug
        if debug:
            self.video_list = video_list[:100]
            if self.include_caption:
                self.caption_list = caption_list[:100]

        self.video_list = video_list
        if self.include_caption:
            self.caption_list = caption_list

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

    def __getitem__(self,idx):
        data_dict = {'dataset': 'pexles'}
        def took_too_long(signum, frame):
            raise TimeoutError('Load', video_path, 'timeout')
        
        while True:
            nidx = self.shuffle_indices[idx]
            try:
                video_path = os.path.join(self.data_dir, self.video_list[nidx])
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

                data_dict['id'] = video_id
                data_dict['folder'] = video_folder
                data_dict['video'] = video.permute(1, 0, 2, 3).contiguous() # c * f * h * w
                data_dict['fps'] = float(self.train_fps)
                data_dict['num_frames'] = data_dict['video'].shape[1]

                if self.include_caption:
                    caption_text = self.caption_list[nidx]
                    data_dict['caption_text'] = caption_text

                return data_dict
                # data_dict: {'dataset': 'pexles', 'id':xxx, 'folder':xxx, 'video':xxx}
            except Exception as e:
                print("Error", e)
                idx = (idx + 1) % self.len_data
                signal.setitimer(signal.ITIMER_REAL, 0)
                continue



def create_pexel_video_dataloader(config):
    train_dataset = PexelsDataset(
        config.dataset.train_meta, 
        config.dataset.data_dir, 
        config.dataset.train_size, 
        config.dataset.resize_list, 
        config.dataset.resize_prob,
        config.dataset.fps_list, 
        config.dataset.random_flip, 
        config.dataset.center_crop
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  
        num_workers=config.dataset.num_workers,      
        pin_memory=False,
    )

    test_dataset = PexelsDataset(
        config.dataset.val_meta, 
        config.dataset.data_dir, 
        config.dataset.train_size, 
        config.dataset.resize_list, 
        config.dataset.resize_prob,
        config.dataset.fps_list, 
        config.dataset.random_flip, 
        config.dataset.center_crop
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  
        num_workers=config.dataset.num_workers,      
        pin_memory=False,
    )
    return train_dataloader, test_dataloader

    

    
        