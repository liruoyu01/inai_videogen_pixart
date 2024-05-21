import pytz
import yaml
from datetime import datetime
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

class Logger():
    def __init__(self, expr_path):
        self.expr_path = expr_path

    def log(self, text, mode='train'):
        msg = '[%s][%s] %s' % (ctime(), mode, text)
        print(msg, flush=True)
        with open('%s/%s_log.txt' % (self.expr_path, mode), 'a') as f:
            f.write('%s\n' % msg)

def ctime():
    ctime = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    return ctime

def tprint(msg):
    ctime = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] %s' % (ctime, msg), flush=True)
    
def save_config(filename, args):
    with open(filename, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)

def save_video_grid(filename, frame, fps, **kwargs):
    f = frame.size(1)
    frame = rearrange(frame, 'b f c h w -> f b c h w')
    frame_list = []
    for i in range(f):
        x = frame[i].squeeze(0)
        x = make_grid(x, **kwargs)
        x = to_pil_image(x)
        frame_list.append(x)
    frame_list[0].save(filename, append_images=frame_list[1:], save_all=True, duration=int(1000/fps), loop=0)
