import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, activation=nn.ReLU(), ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.activation = activation

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / torch.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + self.activation(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, input_size, output_size, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(input_size, input_size, output_size, num_heads, ln=ln)

    def forward(self, x):
        return self.mab(x, x)


class ISAB(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, output_size))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(output_size, input_size, output_size, num_heads, ln=ln)
        self.mab1 = MAB(input_size, output_size, output_size, num_heads, ln=ln)

    def forward(self, x):
        H = self.mab0(self.I.repeat(x.size(0), 1, 1), x)
        return self.mab1(x, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, x):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x)


class SetTransformer(nn.Module):
    def __init__(self, config):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(
                config.x_transf_dim + config.output_dim,
                config.set_transformer_hidden_dim,
                config.set_transformer_num_heads,
                config.set_transformer_num_inds,
                ln=config.set_transformer_ln,
            ),
            ISAB(
                config.set_transformer_hidden_dim,
                config.set_transformer_hidden_dim,
                config.set_transformer_num_heads,
                config.set_transformer_num_inds,
                ln=config.set_transformer_ln,
            ),
        )
        self.dec = nn.Sequential(
            PMA(
                config.set_transformer_hidden_dim,
                config.set_transformer_num_heads,
                config.set_transformer_num_outputs,
                ln=config.set_transformer_ln,
            ),
            SAB(
                config.set_transformer_hidden_dim,
                config.set_transformer_hidden_dim,
                config.set_transformer_num_heads,
                ln=config.set_transformer_ln,
            ),
            SAB(
                config.set_transformer_hidden_dim,
                config.set_transformer_hidden_dim,
                config.set_transformer_num_heads,
                ln=config.set_transformer_ln,
            ),
            nn.Linear(config.set_transformer_hidden_dim, config.hidden_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class DatasetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.dataset_encoder == "set_transformer":
            self.encoder = SetTransformer(config)
        else:
            raise NotImplementedError

    def forward(self, x_context, y_context, x_target):
        xy = torch.cat([x_context, y_context], dim=-1)

        if self.config.dataset_encoder == "set_transformer":
            t = self.encoder(xy)
            # TODO: Check if I need to add mean or sum here too
        return t
