import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CoAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_linear = nn.Linear(input_dim, hidden_dim)
        self.v_linear = nn.Linear(input_dim, hidden_dim)
        self.k_linear = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, v, q):
        q = self.q_linear(q)
        k = self.k_linear(v)
        v = self.v_linear(v)

        # Compute attention weights
        attention_weights = torch.matmul(q, k.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention weights to value vectors
        out = torch.matmul(attention_weights, v)

        # Apply linear layer to output
        out = self.out(out)

        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class CoAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(CoAttention, self).__init__()
#         self.linear1 = nn.Linear(input_dim, input_dim)
#         self.linear2 = nn.Linear(input_dim, input_dim)
#
#     def forward(self, image_feat, audio_feat):
#         image_feat = image_feat.view(image_feat.size(0), image_feat.size(1), -1)
#         audio_feat = audio_feat.view(audio_feat.size(0), audio_feat.size(1), -1)
#
#         image_feat_key = self.linear1(image_feat)
#         audio_feat_key = self.linear2(audio_feat)
#
#         co_attention = F.softmax(torch.bmm(image_feat_key, audio_feat_key.transpose(1, 2)), dim=-1)
#         audio_feat_att = torch.bmm(co_attention, audio_feat)
#         image_feat_att = torch.bmm(co_attention.transpose(1, 2), image_feat)
#
#         return image_feat_att, audio_feat_att

# 使用方法
# co_attention = CoAttention(128)
# image_feat = torch.randn(40, 144, 128)
# audio_feat = torch.randn(40, 144, 128)
# image_feat_att, audio_feat_att = co_attention(image_feat, audio_feat)
