import torch
import torch.nn as nn
import torch.nn.functional as F

from models.knowledge_encoder import KnowledgeEncoder
from models.modules import MLP


class UnderstandingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.knowledge_dim = config.knowledge_dim
        self.knowledge_dropout = config.knowledge_dropout

        if config.knowledge_merge == "sum":
            input_dim = config.hidden_dim
        elif config.knowledge_merge == "concat":
            input_dim = config.hidden_dim + config.knowledge_dim
        elif config.knowledge_merge == "mlp":
            input_dim = config.hidden_dim
            self.knowledge_merger = MLP(
                input_size=config.hidden_dim + config.knowledge_dim,
                hidden_size=config.hidden_dim,
                num_hidden=1,
                output_size=config.hidden_dim,
            )
        else:
            raise NotImplementedError

        if config.use_knowledge:
            self.knowledge_encoder = KnowledgeEncoder(config)
        else:
            self.knowledge_encoder = None

        if config.understanding_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(input_dim, 2 * config.hidden_dim)

        self.config = config

    def forward(self, t, knowledge):
        """
        Obtain an aggregated representation of the dataset and the knowledge, that serves as a form of understanding that is specific to the given task data.
        """
        drop_knowledge = torch.rand(1) < self.knowledge_dropout
        if drop_knowledge or knowledge is None:
            k = torch.zeros((t.shape[0], 1, self.knowledge_dim)).to(t.device)
        else:
            k = self.knowledge_encoder(knowledge)

        if self.config.knowledge_merge == "sum":
            encoder_input = F.relu(t + k)
        elif self.config.knowledge_merge == "concat":
            encoder_input = torch.cat([t, k], dim=-1)
        elif self.config.knowledge_merge == "mlp":
            if knowledge is not None and not drop_knowledge:
                encoder_input = self.knowledge_merger(torch.cat([t, k], dim=-1))
            else:
                encoder_input = F.relu(t)

        u = self.encoder(encoder_input)

        return u

    def get_knowledge_embedding(self, knowledge):
        return self.knowledge_encoder(knowledge).unsqueeze(1)
