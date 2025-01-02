import torch
import torch.nn as nn

from models.modules import MLP


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.decoder_activation == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.GELU()
        self.mlp = MLP(
            input_size=config.hidden_dim + config.x_transf_dim,
            hidden_size=config.decoder_hidden_dim,
            num_hidden=config.decoder_num_hidden,
            output_size=2 * config.output_dim,
            activation=activation,
        )

    def forward(self, x_target, r_final):
        """
        Decode the target set given the target dependent representation

        r_final [num_samples, bs, num_targets, hidden_dim]
        x_target [bs, num_targets, input_dim]
        """
        x_target = x_target.unsqueeze(0).expand(r_final.shape[0], -1, -1, -1)
        xr_target = torch.cat([x_target, r_final], dim=-1)
        p_y_stats = self.mlp(xr_target)

        return p_y_stats
