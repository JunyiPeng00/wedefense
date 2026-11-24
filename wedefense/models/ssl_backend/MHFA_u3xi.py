# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MHFA with U3-xi uncertainty-aware pooling.

Adapted from https://github.com/mrjunjieli/wespeaker_u_cube (Apache-2.0): U_Cube_XI pooling (pooling_layers.py) and
MultiViewTransformerEncoderLayer (MHA.py, here u3xi_mha.py). Adaptation to MHFA:
  * keys / values / layer weights / compression exactly as SSL_BACKEND_MHFA;
  * a frame-level log-precision estimator (Conv1d -> ReLU -> BN -> multi-view transformer layer -> Conv1d ->
    softplus, then 2*log, clamp +-15: identical to U_Cube_XI) runs on the compressed value frames [B, T, D];
  * per head h and value dim d the pooling logits are att_k[t, h] + logprec[t, d] (MHFA's head attention plus the
    U3-xi per-dimension precision), with a learnable prior token (prior_mean, prior_logprec per (h, d), init 0);
  * the Gaussian posterior gives the pooled mean (softmax-weighted values incl. the prior) and the posterior
    variance 1 / sum(exp(logits)) per (h, d);
  * pooling_fc maps the mean to the embedding; the variance is propagated through the same linear map with a diagonal
    approximation var_e = var @ (W ** 2)^T.
forward() returns ONE tensor [B, 2 * embed_dim] = cat(embedding, variance); the projection 'softmax_u3xi' takes the
embedding half.
"""
import torch
import torch.nn as nn

from wedefense.models.ssl_backend.MHFA import GradMultiply
from wedefense.models.ssl_backend.u3xi_mha import MultiViewTransformerEncoderLayer


class SSL_BACKEND_MHFA_U3XI(nn.Module):

    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0,
                 prec_hidden=256,
                 prec_heads=8,
                 prec_dropout=0.2):
        super().__init__()
        self.feature_grad_mult = feature_grad_mult
        self.weights_k = nn.Parameter(torch.ones(nb_layer))
        self.weights_v = nn.Parameter(torch.ones(nb_layer))
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)
        # U3-xi log-precision estimator (U_Cube_XI)
        self.prec_in = nn.Sequential(nn.Conv1d(self.cmp_dim, prec_hidden, kernel_size=1),
                                     nn.ReLU(inplace=True), nn.BatchNorm1d(prec_hidden))
        self.prec_mva = MultiViewTransformerEncoderLayer(embed_dim=prec_hidden, num_heads=prec_heads,
                                                         ff_hidden=prec_hidden * 4, dropout=prec_dropout)
        self.prec_out = nn.Conv1d(prec_hidden, self.cmp_dim, kernel_size=1)
        self.softplus = nn.Softplus(beta=1, threshold=20)
        self.prior_mean = nn.Parameter(torch.zeros(self.head_nb, self.cmp_dim))
        self.prior_logprec = nn.Parameter(torch.zeros(self.head_nb, self.cmp_dim))

    def pool(self, x):
        """x [B, D_in, T, L] -> (pooled mean [B, H*D], posterior variance [B, H*D]); all statistics in float32."""
        x = GradMultiply.apply(x, self.feature_grad_mult)
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)                                   # [B, T, D]
        att_k = self.att_head(k)                                   # [B, T, H]
        with torch.autocast(device_type=v.device.type, enabled=False):
            v32 = v.float()
            h = self.prec_in(v32.transpose(1, 2))                  # [B, hid, T]
            h = self.prec_mva(h.transpose(1, 2)).transpose(1, 2)
            logprec = 2.0 * torch.log(self.softplus(self.prec_out(h)).clamp(min=1e-12))
            logprec = logprec.clamp(-15.0, 15.0).transpose(1, 2)   # [B, T, D]
            logits = att_k.float().unsqueeze(-1) + logprec.unsqueeze(2)   # [B, T, H, D]
            B = v32.shape[0]
            prior_lp = self.prior_logprec.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, -1)
            logits = torch.cat((logits, prior_lp), dim=1)          # [B, T+1, H, D]
            w = torch.softmax(logits, dim=1)
            vals = torch.cat((v32.unsqueeze(2).expand(-1, -1, self.head_nb, -1),
                              self.prior_mean.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, -1)), dim=1)
            mean = (w * vals).sum(dim=1)                           # [B, H, D]
            var = torch.exp(-torch.logsumexp(logits, dim=1))       # 1 / posterior precision
        return mean.flatten(1), var.flatten(1)

    def forward(self, x):
        mean, var = self.pool(x)
        emb = self.pooling_fc(mean.to(self.pooling_fc.weight.dtype))
        with torch.autocast(device_type=mean.device.type, enabled=False):
            var_e = var @ (self.pooling_fc.weight.float() ** 2).t()   # [B, E] diagonal propagation
        return torch.cat((emb.float(), var_e), dim=1)
