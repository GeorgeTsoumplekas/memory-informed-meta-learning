import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import MLP


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.knowledge_merge == "sum":
            input_dim = config.hidden_dim
        elif config.knowledge_merge == "concat":
            input_dim = config.hidden_dim + config.understanding_dim
        elif config.knowledge_merge == "mlp":
            input_dim = config.hidden_dim
            self.knowledge_merger = MLP(
                input_size=config.hidden_dim + config.understanding_dim,
                hidden_size=config.hidden_dim,
                num_hidden=1,
                output_size=config.hidden_dim,
            )
        else:
            raise NotImplementedError

        if config.understanding_aggregator_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.understanding_aggregator_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(input_dim, 2 * config.hidden_dim)

        self.config = config

    def forward(self, r, u):
        if self.config.understanding_merge == "sum":
            encoder_input = F.relu(r + u)

        elif self.config.understanding_merge == "concat":
            encoder_input = torch.cat([r, u], dim=-1)

        elif self.config.understanding_merge == "mlp":
            if u is not None:
                encoder_input = self.understanding_merger(torch.cat([r, u], dim=-1))
            else:
                encoder_input = F.relu(r)

        q_z_stats = self.encoder(encoder_input)

        return q_z_stats
