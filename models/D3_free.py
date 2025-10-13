from transformers import XCLIPVisionModel
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from transformers import XCLIPVisionModel
class D3(nn.Module):
    def __init__(
        self, channel_size=512, dropout=0.2, class_num=1
    ):
        super(D3, self).__init__()
      
        self.backbone = XCLIPVisionModel.from_pretrained("/mnt/sdb/sq/TrainingFreeDet/xclip-base-patch16")
        # self.fc_norm = nn.LayerNorm(768)
        # self.head = nn.Linear(768, 1)
        self.mode = "l2"  # "l2" or "cos"
        self.delta_t = 1.0  # time interval between frames
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def extract_F0(self, x):
        """
        frames: (T, C, H, W) tensor in [0,1] or normalized range.
        Return: (T, D) feature tensor.
        """
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.backbone(images, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)
        
        return F.normalize(sequence_output, dim=-1)  # normalize for cosine stability

    def compute_F1(self, F0):
        """
        Compute first-order temporal difference feature.
        F1_L2(k) = ||F0(k+1) - F0(k)|| / delta_t
        F1_Cos(k) = (1 - cos_sim(F0(k+1), F0(k))) / delta_t
        Return: (T-1,) tensor
        """
        if self.mode.lower() == "l2":
            diffs = torch.norm(F0[:,1:] - F0[:,:-1], dim=-1)
            F1 = diffs / self.delta_t
        elif self.mode.lower() == "cos":
            cos_sim = F.cosine_similarity(F0[:,1:], F0[:,:-1], dim=-1)
            F1 = (1.0 - cos_sim) / self.delta_t
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return F1

    def compute_F2(self, F1):
        """
        Compute second-order central difference.
        F2(k) = (F1(k) - F1(k-1)) / delta_t, for k = 1..len(F1)-1
        Return: (T-2,) tensor
        """
        return (F1[:,1:] - F1[:,:-1]) / self.delta_t

    def volatility_std(self, F2):
        """
        Compute standard deviation σ(F2) as in formula (8):
        σ(F2) = sqrt(1/(T-3) * Σ (F2(i) - mean(F2))²)
        """
        n = F2.numel()
        # print(F2.shape)  # --- IGNORE ---
        if n <= 1:
            return torch.tensor(0.0, device=F2.device)
        mean = torch.mean(F2, dim=-1, keepdim=True)
        # print("sub shape:", sub.shape)  # --- IGNORE ---
        std = torch.sqrt(torch.sum((F2 - mean) ** 2, dim=-1, keepdim=True) / (n - 1))  # (T-3)
        return std

    @torch.no_grad()
    def forward(self, frames):
        """
        frames: (T, C, H, W)
        Return: σ(F2) scalar
        """
        F0 = self.extract_F0(frames)
        # print("F0 shape:", F0.shape)  # --- IGNORE ---
        F1 = self.compute_F1(F0)
        # print("F1 shape:", F1.shape)  # --- IGNORE ---
        F2 = self.compute_F2(F1)
        sigma = self.volatility_std(F2)
        # print("sigma:", sigma)  # --- IGNORE ---

        return 1 - sigma