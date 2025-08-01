# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast


class Extractor(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """

    def __init__(self, attr_nums, backbone='alexnet', dim_chunk=340):
        super(Extractor, self).__init__()

        self.attr_nums = attr_nums
        if backbone == 'alexnet':
            self.backbone = torchvision.models.alexnet(weights='DEFAULT')
            self.backbone.classifier = self.backbone.classifier[:-2]
            dim_init = 4096
        if backbone == 'resnet':
            self.backbone = torchvision.models.resnet18(weights='DEFAULT')
            self.backbone.fc = nn.Sequential()
            dim_init = 512
        if backbone == 'vit':
            self.backbone = torchvision.models.vit_b_16(weights='DEFAULT')
            self.backbone.heads = nn.Sequential()
            dim_init = 768

        dis_proj = []
        for i in range(len(attr_nums)):
            dis_proj.append(nn.Sequential(
                nn.Linear(dim_init, dim_chunk),
                nn.ReLU(),
                nn.Linear(dim_chunk, dim_chunk)
            )
            )
        self.dis_proj = nn.ModuleList(dis_proj)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(nn.Sequential(
                nn.Linear(dim_chunk, attr_nums[i]))
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    @autocast()
    def forward(self, img):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        feat = self.backbone(img)
        dis_feat = []
        for layer in self.dis_proj:
            dis_feat.append(layer(feat))

        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(dis_feat[i]).squeeze())
        return dis_feat, attr_classification_out


class MemoryBlock(nn.Module):
    """
    Store the prototype embeddings of all attribute values
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        dim_chunk: int, the size of each attribute-specific embedding
    """

    def __init__(self, attr_nums, dim_chunk=340):
        super(MemoryBlock, self).__init__()
        self.Memory = nn.Linear(np.sum(attr_nums), len(attr_nums) * dim_chunk, bias=False)

    @autocast()
    def forward(self, indicator):
        t = self.Memory(indicator)
        return t


class Extractor_AP(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """

    def __init__(self, attr_nums, backbone='vit', dim_chunk=340):
        super(Extractor_AP, self).__init__()

        self.attr_nums = attr_nums
        self.backbone = torchvision.models.vit_b_16()
        self.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
            url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            map_location=torch.device('cpu')
        ))
        self.backbone.heads = nn.Sequential()

        self.hidden_dim = self.backbone.hidden_dim
        self.dim_chunk = dim_chunk
        self.attr_queries = nn.Parameter(torch.randn(len(attr_nums), self.dim_chunk))
        self.attention_pooler = nn.MultiheadAttention(self.dim_chunk, 10, kdim=self.hidden_dim,
                                                      vdim=self.hidden_dim, bias=False, batch_first=True)
        self.ln = nn.LayerNorm(self.dim_chunk, eps=1e-6)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(nn.Sequential(
                nn.Linear(self.dim_chunk, attr_nums[i]))
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    def forward(self, img):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        x = self.backbone._process_input(img)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)  # (batch, 196+1, 768)
        q = einops.repeat(self.attr_queries, 'l d -> n l d', n=n)
        attr_token, _ = self.attention_pooler(query=q, key=x, value=x, need_weights=False)
        attr_token = self.ln(attr_token)
        dis_feat = torch.chunk(attr_token, chunks=self.dim_chunk, dim=1)
        dis_feat = [f.squeeze(1) for f in dis_feat]

        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(dis_feat[i]))
        return dis_feat, attr_classification_out
