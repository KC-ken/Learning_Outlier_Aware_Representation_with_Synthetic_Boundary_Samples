"""
https://github.com/HobbitLong/SupContrast
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from utils import synthesize_OOD


class VOConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, vos_mode="Cont", contrast_mode="vos", base_temperature=0.07, lamb = 1):
        super(VOConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.vos_mode = vos_mode
        self.lamb = lamb
        # self.resample = resample
        # self.near_OOD = near_OOD

    def forward(self, features, labels=None, negative_features=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("unbind: ", torch.unbind(features, dim=1))
        # print("fs: ", features.size())
        # print("con size: ", contrast_feature.size())
        
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == "vos":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        log_prob = (mask * logits).sum(1)

        if negative_features is not None:
            # compute negative logits
            negative_dot_contrast = torch.div(
                torch.matmul(anchor_feature, negative_features.T), self.temperature
            )

            # for numerical stability
            n_logits_max, _ = torch.max(negative_dot_contrast, dim=1, keepdim=True)
            n_logits = negative_dot_contrast - n_logits_max.detach()

            n_exp_logits = (torch.exp(n_logits) * logits_mask).sum(1)
        else:
            n_exp_logits = 0
        #------------------------------------------------------------------

        # compute log_prob = - L_cont
        exp_logits = (torch.exp(logits) * logits_mask).sum(1)

        if self.vos_mode == "Cont":
            log_prob -= torch.log(exp_logits + self.lamb * n_exp_logits)
        elif self.vos_mode == "DualCont":
            # log_prob += torch.log(1 / exp_logits + self.lamb / n_exp_logits) #dual
            log_prob -= (torch.log(exp_logits) + self.lamb * torch.log(n_exp_logits)) #dual_out
        else:
            raise ValueError("Unknown vos_mode: {}".format(self.vos_mode))
        # print("\n\nafter:", log_prob.size())
        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
