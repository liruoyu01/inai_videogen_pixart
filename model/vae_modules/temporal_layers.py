from einops import rearrange

import torch
from torch import nn


class TemporalConvBlock(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, dropout=0.0, up_sample=False, down_sample=False, spa_stride=1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        spa_stride = spa_stride
        spa_pad = int((spa_stride-1)*0.5)
        # conv layers
        # self.conv1 = nn.Sequential(
        #     nn.GroupNorm(32, in_dim), 
        #     nn.SiLU(), 
        #     nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        # )
        if down_sample:
            self.conv1 = nn.Sequential(
                nn.GroupNorm(32, in_dim), 
                nn.SiLU(), 
                nn.Conv3d(in_dim, out_dim, (3, spa_stride, spa_stride), stride=(2,1,1), padding=(1, spa_pad, spa_pad))
            )
        else:
            self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), 
            nn.SiLU(), 
            nn.Conv3d(in_dim, out_dim, (3, spa_stride, spa_stride), padding=(1, spa_pad, spa_pad))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(1, spa_pad, spa_pad)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(1, spa_pad, spa_pad)),
        )
        if up_sample:
            self.conv4 = nn.Sequential(
                nn.GroupNorm(32, out_dim), 
                nn.SiLU(), 
                nn.Conv3d(out_dim, in_dim*2, (3, spa_stride, spa_stride), padding=(1, spa_pad, spa_pad))
            )
        # elif down_sample:
        #     self.conv4 = nn.Sequential(
        #         nn.GroupNorm(32, out_dim),
        #         nn.SiLU(),
        #         nn.Dropout(dropout),
        #         nn.Conv3d(out_dim, in_dim, (3, 1, 1), stride=(2,1,1), padding=(1, 0, 0)),
        #     )
        else:
            self.conv4 = nn.Sequential(
                nn.GroupNorm(32, out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv3d(out_dim, in_dim, (3, spa_stride, spa_stride), padding=(1, spa_pad, spa_pad)),
            )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

        self.down_sample = down_sample
        self.up_sample = up_sample
        # if down_sample:
        #     self.temp_pooling = nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1))
            # self.temp_pooling = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))
        # elif up_sample:
        #     self.temp_upsampler = nn.Upsample(scale_factor=(2.0, 1.0, 1.0), mode='trilinear')

    def forward(self, hidden_states, num_frames=1):
        hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w', f=num_frames)
        identity = hidden_states

        if self.down_sample:
            # identity = self.temp_pooling(identity)
            identity = identity[:,:,::2]
            # identity = identity[:,:,::2] * 0.5 + identity[:,:,1::2] * 0.5
        elif self.up_sample:
            hidden_states_new = torch.cat((hidden_states,hidden_states),dim=2)
            hidden_states_new[:, :, 0::2] = hidden_states
            hidden_states_new[:, :, 1::2] = hidden_states
            identity = hidden_states_new
            del hidden_states_new

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        if self.up_sample:
            # o2 = rearrange(o2, 'b (d c) f h w -> b c (f d) h w', d=2)
            hidden_states = rearrange(hidden_states, 'b (d c) f h w -> b c (f d) h w', d=2)

        hidden_states = identity + hidden_states

        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')
        return hidden_states