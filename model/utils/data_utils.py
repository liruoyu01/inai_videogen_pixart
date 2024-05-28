import cv2
from typing import List
from torchvision import transforms as T
import tempfile
import torch
import imageio

def export_to_video(
    video_frames: List[torch.Tensor], 
    output_video_path: str = None, 
    fps: int = 10,
) -> str:
    # each frame shape (h, w, c)
    try:
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

        video_frames = [(frame * 255).clamp(0, 255).to(dtype=torch.uint8) for frame in video_frames]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w, c = video_frames[0].shape

        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
        for frame in video_frames:
            img = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        return output_video_path
    
    except Exception as e:
        print('error ocured during saving', e)
        return ''

def export_to_gif(
    video_frames: List[torch.Tensor],
    output_video_path: str=None,
    duration: float=1.0, # in second
    loop: int=0,
    fps: int=30,
):
    # each frame shape (h, w, c) 
    # frame pixel value [0,1]
    try:
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix=".gif").name
        # gif pixel range [0,255]
        # dtype=uint8
        # imageio only works with numpy ndarray
        video_frames = [(frame * 255).to(dtype=torch.uint8).numpy() for frame in video_frames]
        imageio.mimwrite(output_video_path, video_frames, loop=loop, duration=duration, fps=fps)

    except Exception as e:
        print('error ocured during saving', e)
        return ''