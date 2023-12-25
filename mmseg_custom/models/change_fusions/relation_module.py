import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from mmseg.registry import MODELS
"""RCAM"""
@MODELS.register_module()
class RCAM(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0, head_dim=32):
        super(OCA, self).__init__()
        self.n = nhead
        self.head = nn.ModuleList()
        for i in range(self.n):
            self.head.append(OCA(d_model, head_dim))
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query=None, key=None, value=None, pos_1=None, pos_2=None):
        query = self.with_pos_embed(query, pos_1)
        key = self.with_pos_embed(key, pos_2)
        value = self.with_pos_embed(value, pos_2)
        for i in range(self.n):
            if i == 0:
                concat = self.head[i](query, key, value)
            else:
                concat = torch.cat((concat, self.head[i](query, key, value)), -1)
        y = concat
        return y

"""OCA"""
class OCA(nn.Module):
    def __init__(self, in_dim=256, mid_dim=32):
        super(OCA, self).__init__()
        self.temperature = 1e-9
        self.K = nn.Linear(in_dim, mid_dim, bias=False)
        self.Q = nn.Linear(in_dim, mid_dim, bias=False)
        self.V = nn.Linear(in_dim, in_dim, bias=False)
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.K.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.Q.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.V.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        x2_k = self.K(key)
        x2_k = F.normalize(x2_k, p=2, dim=-1)
        x2_k = x2_k.permute(1, 2, 0)

        x1_q = self.Q(query)
        x1_q = F.normalize(x1_q, p=2, dim=-1)
        x1_q = x1_q.permute(1, 0, 2)

        corr = F.softmax(torch.bmm(x1_q, x2_k), dim=-1) 
        corr = corr / (self.temperature + corr.sum(dim=1, keepdim=True))

        x2_v = self.V(value)
        x2_v = x2_v.permute(1, 0, 2)
        
        y = torch.bmm(corr, x2_v).permute(1, 0, 2)

        y = self.linear(query - y)

        return F.relu(y)