import torch
import torch.nn as nn


class DataInteractionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.data_interaction_self_attention_hidden_dim,
            num_heads=config.data_interaction_self_attention_num_heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.data_interaction_cross_attention_hidden_dim,
            num_heads=config.data_interaction_cross_attention_num_heads,
            batch_first=True,
        )

    def forward(self, x_context, y_context, x_target):
        xy = torch.cat([x_context, y_context], dim=-1)

        rs = self.self_attention(xy, xy, xy)  # TODO: Maybe need to add [0]?
        rs = self.cross_attention(x_target, xy, rs)[0]

        r = torch.mean(rs, dim=1, keepdim=True)

        return r
