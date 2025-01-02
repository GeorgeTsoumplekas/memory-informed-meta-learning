import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_hidden, output_size, activation=nn.GELU()
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            (
                [nn.Linear(input_size, hidden_size)]
                + [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)]
                + [nn.Linear(hidden_size, output_size)]
            )
        )
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class XEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_hidden=config.x_encoder_num_hidden,
            output_size=config.x_transf_dim,
        )

    def forward(self, x):
        return self.mlp(x)


class XYEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pairer = MLP(
            input_size=config.x_transf_dim + config.output_dim,
            hidden_size=config.xy_encoder_hidden_dim,
            num_hidden=config.xy_encoder_num_hidden,
            output_size=config.hidden_dim,
        )
        if config.data_agg_func == "cross-attention":
            self.cross_attention = MultiheadAttention(
                config.hidden_dim,
                num_heads=4,
                batch_first=True,
            )

        self.config = config

    def forward(self, x_context, y_context, x_target):
        """
        Encode the context set all together
        """
        xy = torch.cat([x_context, y_context], dim=-1)
        Rs = self.pairer(xy)
        # aggregate
        if self.config.data_agg_func == "mean":
            R = torch.mean(Rs, dim=1, keepdim=True)  # [bs, 1, r_dim]
        elif self.config.data_agg_func == "sum":
            R = torch.sum(Rs, dim=1, keepdim=True)
        elif self.config.data_agg_func == "cross-attention":
            Rs = self.cross_attention(x_target, x_context, Rs)[0]
            R = torch.mean(Rs, dim=1, keepdim=True)
        return R


class LatentEncoder(nn.Module):
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

        if config.latent_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(input_dim, 2 * config.hidden_dim)
        self.config = config

    def forward(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """
        drop_knowledge = torch.rand(1) < self.knowledge_dropout
        if drop_knowledge or knowledge is None:
            k = torch.zeros((R.shape[0], 1, self.knowledge_dim)).to(R.device)

        else:
            k = self.knowledge_encoder(knowledge)

        if self.config.knowledge_merge == "sum":
            encoder_input = F.relu(R + k)

        elif self.config.knowledge_merge == "concat":
            encoder_input = torch.cat([R, k], dim=-1)

        elif self.config.knowledge_merge == "mlp":
            if knowledge is not None and not drop_knowledge:
                encoder_input = self.knowledge_merger(torch.cat([R, k], dim=-1))
            else:
                encoder_input = F.relu(R)

        q_z_stats = self.encoder(encoder_input)

        return q_z_stats

    def get_knowledge_embedding(self, knowledge):
        return self.knowledge_encoder(knowledge).unsqueeze(1)


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

    def forward(self, x_target, R_target):
        """
        Decode the target set given the target dependent representation

        R_target [num_samples, bs, num_targets, hidden_dim]
        x_target [bs, num_targets, input_dim]
        """
        x_target = x_target.unsqueeze(0).expand(R_target.shape[0], -1, -1, -1)
        XR_target = torch.cat([x_target, R_target], dim=-1)
        p_y_stats = self.mlp(XR_target)
        return p_y_stats