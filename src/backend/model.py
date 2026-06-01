import os

import numpy as np
import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class OptimizedCond3DUNet(nn.Module):
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1, base_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim),
        )
        self.conv_in = nn.Conv3d(in_channels + cond_channels, base_dim, 3, padding=1)
        self.down = nn.Sequential(
            nn.Conv3d(base_dim, base_dim * 2, 4, stride=2, padding=1),
            ResBlock3D(base_dim * 2),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose3d(base_dim * 2, base_dim, 4, stride=2, padding=1),
            ResBlock3D(base_dim),
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(4, base_dim),
            nn.SiLU(),
            nn.Conv3d(base_dim, out_channels, 3, padding=1),
        )

    def forward(self, x_t, t, condition):
        x = torch.cat([x_t, condition], dim=1)
        x = self.conv_in(x)
        t_emb = self.time_embed(t).view(-1, x.shape[1], 1, 1, 1)
        x = x + t_emb
        x_down = self.down(x)
        x_up = self.up(x_down)
        return self.conv_out(x_up)


class FlowMatchingModel:
    def __init__(self, net: nn.Module, device: torch.device):
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def sample_ode(self, condition, steps=50, save_frames_dir=None, save_every=2):
        batch_size, _, depth, height, width = condition.shape
        x_t = torch.randn(batch_size, 1, depth, height, width, device=self.device)
        dt = 1.0 / steps

        for step in range(steps):
            t = torch.full((batch_size, 1), step / steps, device=self.device)
            v_pred = self.net(x_t, t, condition)
            x_t = x_t + v_pred * dt

            if save_frames_dir is not None and step % save_every == 0:
                frame = torch.round(x_t).cpu().squeeze().numpy().astype(np.int16)
                np.save(os.path.join(save_frames_dir, f"frame_{step:03d}.npy"), frame)

        if save_frames_dir is not None:
            frame = torch.round(x_t).cpu().squeeze().numpy().astype(np.int16)
            np.save(os.path.join(save_frames_dir, f"frame_{steps:03d}.npy"), frame)

        return x_t


def load_flow_matching_model(model_path: str, device: torch.device) -> FlowMatchingModel:
    net = OptimizedCond3DUNet(in_channels=1, cond_channels=3, out_channels=1, base_dim=32)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return FlowMatchingModel(net, device=device)
